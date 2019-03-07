import numpy as np
import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import time
import math
import sys
import json
import errno


class DataProcessor(object):

    """Extracts data from given python-friendly formatted dataset.

    Parameters
    ----------
    path : str
        Path to data directory. Must contain a spikes subdirectory with spikes in ms.
    time_info : TimeInfo
        Object that holds timing information including the beginning and
        end of the region of interest and the time bin. All in milliseconds.
    cell_range : range
        Beginning and end cell to be analyzed, in range format.
    num_conditions : int
        Integer signifying the number of experimental conditions in the
        dataset. Conditions are information that exists on a per trial basis.

    Attributes
    ----------
    path : str
        Path to data directory. Must contain a spikes subdirectory with spikes in ms.
    time_info : TimeInfo
        Object that holds timing information including the beginning and
        end of the region of interest and the time bin. All in milliseconds.
    cell_range : range
        Beginning and end cell to be analyzed, in range format.
    num_conditions : int
        Integer signifying the number of experimental conditions in the
        dataset.
    spikes : numpy.ndarray
        Array of spike times in milliseconds, of dimension (trials × time).
    num_trials : numpy.ndarray
        Array containing integers signifying the number of trials a given cell has data for.
    spikes_summed : dict (int: numpy.ndarray)
        Dict containing summed spike data for all cells, indexed by cell.
    spikes_binned : dict (int: numpy.ndarray)
        Dict containing binned spike data for all cells, indexed by cell.
    spikes_summed_cat : dict (int: numpy.ndarray)
        Dict containing binned spike data for all cells, indexed by cell and category.
    conditions_dict : dict (tuple of int, int: numpy.ndarray of int)
        Dict containing condition information for all cells.
        Indexed by cell and condition.

    """

    def __init__(self, path, cell_range, time_info=None):
        self.path = path
        self.cell_range = cell_range
        self.spikes = self._extract_spikes()
        self.num_trials = self._extract_num_trials()
        conditions = self._extract_conditions()
        if conditions is not None:
            # this will fail if first cell doenst include all cond, hack
            self.num_conditions = len(np.unique(conditions[0]))
        self.conditions_dict = self._associate_conditions(conditions)
        # if time_info is not provided, a default window will be constructed
        # based off the min and max values found in the data
        if time_info is None:
            self.time_info = self._set_default_time()
        else:
            self.time_info = time_info

        self.spike_info = self.extract_spike_info()
        self.spikes_binned = self._bin_spikes()
        self.spikes_summed = self._sum_spikes()
        self.spikes_summed_cat = self._sum_spikes_conditions(conditions)

    def _set_default_time(self):
        """Parses spikes and finds min and max spike times for bounds.

        """
        max_time = sys.float_info.min
        min_time = sys.float_info.max
        for cell in self.spikes:
            for trial in self.spikes[cell]:
                for t in trial:
                    if t > max_time:
                        max_time = t
                    if t < min_time:
                        min_time = t
        return [min_time, max_time]

    def _extract_spikes(self):
        """Extracts spike times from data file.

        Returns
        -------
        dict (int: numpy.ndarray of float?)
            Contains per cell spike times.

        """
        spikes = {}
        if os.path.exists(self.path + "/spikes/"):
            for i in self.cell_range:
                spike_path = self.path + '/spikes/%d.json' % i
                with open(spike_path, 'rb') as f:
                    spikes[i] = np.array(json.load(f, encoding="bytes"))
        else:
            print("Spikes folder not found.")
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), self.path+"/spikes/")
        return spikes

    def _extract_num_trials(self):
        """Extracts number of trials per cell.

        Returns
        -------
        numpy.ndarray of int
            Array of dimension [Number of cells] that provides the number of
            trials of a given cell.

        """
        if os.path.exists(self.path + "/number_of_trials.json"):
            with open(self.path + "/number_of_trials.json", 'rb') as f:
                return np.array(json.load(f, encoding="bytes"))
        else:
            raise FileNotFoundError("number_of_trials.json not found")

    def _extract_conditions(self):
        """Extracts trial conditions per cell per trial.

        Returns
        -------
        numpy.ndarray of int
            Array of dimension [Number of cells] × [Trials] that provides the condition
            for the trial.

        """
        # convert keys to int
        if os.path.exists(self.path + "/conditions.json"):
            with open(self.path + "/conditions.json", 'rb') as f:

                return {int(k): v for k, v in json.load(f, encoding="bytes").items()}

        else:
            print("conditions.json not found")
            return None

    def extract_spike_info(self):
        if os.path.exists(self.path+"/spike_info.json"):
            with open(self.path + "/spike_info.json", 'rb') as f:
                data = json.load(f)

            return data

        else:
            print("spike_info not found")
            return None

    def _sum_spikes(self):
        """Sums spike data over trials.

        Returns
        -------
        dict (int: numpy.ndarray of int)
            Summed spike data.

        """
        spikes = self.spikes_binned
        summed_spikes = {}
        for cell in self.cell_range:
            summed_spikes[cell] = np.sum(spikes[cell], 0)
        return summed_spikes

    def _sum_spikes_conditions(self, conditions):
        """Sums spike data over trials per condition

        Parameters
        ----------
        conditions : numpy.ndarray of int
            Array of dimension [Number of cells] × [Trials] that provides the condition
            for the trial.

        Returns
        -------
        numpy.ndarray of int
            Array of dimension [Cells] × [Condition] × [Time].

        """
        spikes = self.spikes_binned

        if conditions is None:
            return None
        else:
            summed_spikes_condition = {}
            for cell in self.cell_range:
                summed_spikes_condition[cell] = {}
                for condition in range(self.num_conditions):
                    summed_spikes_condition[cell][condition+1] = {}
                    summed_spikes_condition[cell][condition+1] = np.sum(
                        spikes[cell].T * self.conditions_dict[cell][condition + 1].T, 1)

            return summed_spikes_condition

    def _bin_spikes(self):
        """Bins spikes within the given time range into 1 ms bins.

        """
        time_low, time_high = self.time_info[0], self.time_info[1]
        total_bins = time_high - time_low
        spikes_binned = {}
        for cell in self.spikes:
            spikes_binned[cell] = np.zeros(
                (self.num_trials[cell], total_bins))
            for trial_index, trial in enumerate(self.spikes[cell]):
                if type(trial) is float or type(trial) is int:
                    trial = [trial]
                for value in trial:
                    if value <  time_high and value >= time_low:
                        spikes_binned[cell][trial_index][int(
                            value - time_low)] = 1

        return spikes_binned

    def _associate_conditions(self, conditions):
        """Builds dictionary that associates trial and condition.

        Returns
        -------
        dict (int, int: np.ndarray of int)
            Dict indexed by condition AND cell number returning array of trials.
            Array contains binary data: whether or not the trial is of the indexed condition.

        """
        if conditions is None:
            return None
        else:
            conditions_dict = {}
            for cell in self.cell_range:
                conditions_dict[cell] = {
                    i+1: np.zeros((self.num_trials[cell], 1)) for i in range(self.num_conditions)}
                cond = conditions[cell][0:self.num_trials[cell]]
                for trial, condition in enumerate(cond):
                    if condition:
                        conditions_dict[cell][condition][trial] = 1
            return conditions_dict

    def save_attribute(self, attribute, filename, path=""):
        """Saves data_processor attribute to disk.

        """
        with open((os.getcwd() + path + "/{0}.json").format(filename), 'w') as f:
            json.dump(attribute, f)