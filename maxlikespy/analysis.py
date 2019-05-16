import matplotlib as mpl
import os
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import maxlikespy.plotting as cellplot
import maxlikespy.util as util
import models as models
import time
import numpy as np
import json
import sys
from scipy.stats import chi2
import matplotlib.pyplot as plt
import errno


class DataProcessor(object):

    """Extracts data from given python-friendly formatted dataset.

    Parameters
    ----------
    path : str
        Path to data directory. Must contain a spikes subdirectory with spikes in ms.
    window : ndarray
        Array that holds timing information including the beginning and
        end of the region of interest and the time bin. All in milliseconds.
        If not supplied, it is assumed that trial lengths are unequal and will be loaded from file.
    cell_range : range
        Beginning and end cell to be analyzed, in range format.

    Attributes
    ----------
    path : str
        Path to data directory. Must contain a spikes subdirectory with spikes in ms.
    window : ndarray
        Array that holds timing information including the beginning and
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

    def __init__(self, path, cell_range, window=None):
        self._check_results_dir()
        self.path = path
        self.cell_range = cell_range
        self.spikes = self._extract_spikes()
        self.num_trials = self._extract_num_trials()
        conditions = self._extract_conditions()
        if conditions is not None:
            # finds total number of different conditions in supplied file
            self.num_conditions = len(
                set([item for sublist in list(conditions.values()) for item in sublist]))
        self.conditions_dict = self._associate_conditions(conditions)
        # if window is not provided, a default window will be constructed
        # based off the min and max values found in the data
        if window and len(window) == 2:
            print("Time window provided. Assuming all trials are of equal length")
            min_time = np.full(max(self.num_trials), window[0])
            max_time = np.full(max(self.num_trials), window[1])
            self.window = np.array(list(zip(min_time, max_time)))
        elif not window:
            self.window = self._extract_trial_lengths()
        # if self.window is None:
        #     self.window = self._set_default_time()
        self.spike_info = self._extract_spike_info()
        self.spikes_binned = self._bin_spikes()
        self.spikes_summed = self._sum_spikes()
        self.spikes_summed_cat = self._sum_spikes_conditions(conditions)

    def _check_results_dir(self):
        """Creates directories for artifacts if they don't exist.

        """
        if not os.path.exists(os.getcwd()+"/results"):
            os.mkdir(os.getcwd()+"/results")
            os.mkdir(os.getcwd()+"/results/figs")

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

    def _extract_trial_lengths(self):
        """Extracts trial lengths from file.

        """
        path = self.path + "/trial_lengths.json"
        try:
            with open(path, 'r') as f:
                return np.array(json.load(f))
        except:
            print("trial_lengths.json not found")
            return None

    def _extract_spikes(self):
        """Extracts spike times from data file.

        Returns
        -------
        dict (int: numpy.ndarray of float)
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
                loaded = json.load(f, encoding="bytes")
                if type(loaded) is dict:
                    return {int(k): v for k, v in loaded.items()}
                if type(loaded) is list:
                    return {int(k): v for k, v in enumerate(loaded)}

        else:
            print("conditions.json not found")

            return None

    def _extract_spike_info(self):
        """Extracts spike info from file.

        """
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
            summed_spikes[cell] = np.nansum(spikes[cell], 0)

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
        lower_bounds, upper_bounds = self.window[:, 0], self.window[:, 1]
        total_bins = int(max(upper_bounds) - min(lower_bounds))
        spikes_binned = {}
        for cell in self.spikes:
            spikes_binned[cell] = np.zeros(
                (int(self.num_trials[cell]), total_bins))
            for trial_index, trial in enumerate(self.spikes[cell]):
                time_low, time_high = lower_bounds[trial_index], upper_bounds[trial_index]
                # total_bins = time_high - time_low
                if type(trial) is float or type(trial) is int:
                    trial = [trial]
                for value in trial:
                    if value < time_high and value >= time_low:
                        spikes_binned[cell][trial_index][int(
                            value - time_low)] = 1
                if trial_index < self.num_trials[cell]:
                    spikes_binned[cell][trial_index][upper_bounds[trial_index]:] = np.nan

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

class Pipeline(object):

    """Performs fitting procedure and model comparison for all cells.

    Parameters
    ----------
    cell_range : range
        Beginning and end cell to be analyzed, in range format.
    data_processor : DataProcessor
        Object returned by data processing module that includes all relevent cell data.
    models : list of str
        List of model names as strings to be used in analysis.
        These must match the names used in "fit_x" methods.
    subsample : float
        Number signifying the percentage of trials to be used, 0 will use all trials.

    Attributes
    ----------
    time_start : float
        Time keeping variable for diagnostic purposes.
    cell_range : range
        Beginning and end cell to be analyzed, as a Range
    data_processor : DataProcessor
        Object returned by data processing module that includes all relevent cell data.
    subsample : float
        Number signifying the percentage of trials to be used, 0 will use all trials.
    model_dict : dict of Model
        Dict that contains all initialized Model instances.

    """

    def __init__(self, cell_range, data_processor, models, subsample=0, save_dir=None):
        self.time_start = time.time()
        self.cell_range = cell_range
        self.data_processor = data_processor
        self.subsample = subsample
        self.model_dict = self._make_models(models)
        self.save_dir = save_dir

    def _make_models(self, models_to_fit, even_odd=None):
        """Initializes and creates dict of models to fit.

        """
        model_dict = {model: {} for model in models_to_fit}
        condition_data = self.data_processor.conditions_dict
        spike_info = self.data_processor.spike_info
        for cell in self.cell_range:
            num_trials, spikes_binned, conditions = self._select_model_data(
                cell)
            for model in models_to_fit:

                # data to be passed to models
                model_data = {}
                model_data['spikes'] = spikes_binned
                model_data['window'] = self.data_processor.window
                model_data['num_trials'] = num_trials
                if condition_data:
                    model_data['conditions'] = conditions
                if spike_info:
                    model_data['spike_info'] = spike_info[str(cell)]
                # this creates an instance of class "model" in the module "models"
                model_instance = getattr(models, model)(model_data)
                model_dict[model][cell] = model_instance

        return model_dict

    def _select_model_data(self, cell):
        """Provides important data for model construction.
        If Subsampling, data will also be subsampled.

        """
        condition_data = self.data_processor.conditions_dict
        if self.subsample:

            sampled_trials = self._subsample_trials(
                self.data_processor.num_trials[cell], self.subsample)
            num_trials = len(sampled_trials)
            spikes_binned = self.data_processor.spikes_binned[cell][sampled_trials, :]
            if condition_data:
                conditions = {cond: condition_data[cell][cond][sampled_trials]
                              for cond in condition_data[cell]}
        else:
            num_trials = self.data_processor.num_trials[cell]
            spikes_binned = self.data_processor.spikes_binned[cell]
            if condition_data:
                conditions = condition_data[cell]
            else:
                conditions = None

        return num_trials, spikes_binned, conditions

    def _subsample_trials(self, num_trials, subsample):
        """Randomly selects a percentage of total trials

        """
        num_trials = int(num_trials * subsample)
        if num_trials < 1:
            num_trials = 1
        sampled_trials = np.random.randint(
            num_trials,
            size=num_trials)

        return sampled_trials

    def set_model_bounds(self, models, bounds):
        """Sets solver bounds for the given model for all cells.

        Parameters
        ----------
        models : string
            String with same name as desired model.

        bounds : dict of list or list of dicts
            Dict of lower and upper bounds for the solver, including param names.

        """
        if type(models) is list:
            assert len(models) == len(bounds)
            for cell in self.cell_range:
                for index, model in enumerate(models):
                    if model in self.model_dict:
                        self.model_dict[model][cell].set_bounds(bounds[index])
                    else:
                        raise ValueError("model does not match supplied models")
        else:
            for cell in self.cell_range:
                if models in self.model_dict:
                    self.model_dict[models][cell].set_bounds(bounds)
                else:
                    raise ValueError("model does not match supplied models")

    def set_model_x0(self, models, x0):
        """Sets model x0 for the given model for all cells.

        Parameters
        ----------
        models : string
            String with same name as desired model.

        x0 : list or list of lists
            List of initial state for given model params.

        """
        if type(models) is list:
            assert len(models) == len(x0)
            for cell in self.cell_range:
                for index, model in enumerate(models):
                    if model in self.model_dict:
                        self.model_dict[model][cell].set_x0(x0[index])
                    else:
                        raise ValueError("model does not match supplied models")
        else:
            for cell in self.cell_range:
                if models in self.model_dict:
                    self.model_dict[models][cell].set_x0(x0)
                else:
                    raise ValueError("model does not match supplied models")

    def set_model_info(self, model, name, data, per_cell=False):
        """Sets model info attribute for all cells.

        Parameters
        ----------
        model : string
            String with same name as desired model.
        name : string
            Name for supplied data.
        data : hashable object
            Data to be supplied to model for processing.
        per_cell : bool
            Flag to be set true if input data has a per-cell structure.

        """
        for cell in self.cell_range:
            if model in self.model_dict:
                if per_cell:
                    self.model_dict[model][cell].set_info(name, data[cell])
                else:
                    self.model_dict[model][cell].set_info(name, data)
            else:
                raise ValueError("model does not match supplied models")

    def _get_even_odd_trials(self, cell, even):
        """Returns either even or odd subset of binned spikes.

        """
        if even:
            return self.data_processor.spikes_binned[cell][::2]
        else:
            return self.data_processor.spikes_binned[cell][1::2]

    def fit_even_odd(self, solver_params=None):
        """Fits even and odd trial parameters for all models then saves to disk.

        Parameters
        ----------
        solver_params : dict
            Dict of parameters for minimizer algorithm.

        """
        if self.subsample:
            print("Splitting trials while also subsampling may lead to unsatisfactory results")

        fits_even, fits_odd = {}, {}
        lls_even, lls_odd = {}, {}
        for cell in self.cell_range:
            print("Fitting cell {0}".format(cell))
            fits_even[cell] = {}
            fits_odd[cell] = {}
            lls_even[cell], lls_odd[cell] = {}, {}
            for model in self.model_dict:
                model_instance = self.model_dict[model][cell]
                if model_instance.bounds is None:
                    raise ValueError(
                        "model \"{0}\" bounds not yet set".format(model)
                    )
                if model_instance.x0 is None:
                    raise ValueError(
                        "model \"{0}\" x0 not yet set".format(model)
                    )
                # Even trials
                model_instance.spikes = self._get_even_odd_trials(cell, True)
                print("fitting {0}".format(model))
                model_instance.fit_params(solver_params)

                # Build dict for json dump, json requires list instead of ndarray
                param_dict = {param: model_instance.fit.tolist()[index]
                              for index, param in enumerate(model_instance.param_names)}
                fits_even[cell][model_instance.__class__.__name__] = param_dict
                lls_even[cell][model_instance.__class__.__name__] = model_instance.fun

                # Odd trials
                model_instance.spikes = self._get_even_odd_trials(cell, False)
                print("fitting {0}".format(model))
                model_instance.fit_params(solver_params)
                param_dict = {param: model_instance.fit.tolist()[index]
                              for index, param in enumerate(model_instance.param_names)}
                fits_odd[cell][model_instance.__class__.__name__] = param_dict
                lls_odd[cell][model_instance.__class__.__name__] = model_instance.fun

            util.save_data({cell:fits_even[cell]}, "cell_fits_even", path=self.save_dir, cell=cell)
            util.save_data({cell:lls_even[cell]}, "log_likelihoods_even", path=self.save_dir, cell=cell)

            util.save_data({cell:fits_odd[cell]}, "cell_fits_odd", path=self.save_dir, cell=cell)
            util.save_data({cell:lls_odd[cell]}, "log_likelihoods_odd", path=self.save_dir, cell=cell)

    def fit_all_models(self, solver_params=None):
        """Fits parameters for all models then saves to disk.

        Parameters
        ----------
        solver_params : dict
            Dict of parameters for minimizer algorithm.

        """
        if not solver_params:
            print("Solver params not set, using preconfigured defaults")
            solver_params = {
                "niter": 200,
                "stepsize": 100,
                "interval": 10,
                "method": "TNC",
                "use_jac": False,
            }
    
        cell_fits = {}
        cell_lls = {}

        for cell in self.cell_range:
            print("Fitting cell {0}".format(cell))
            cell_fits[cell] = {}
            cell_lls[cell] = {}
            for model in self.model_dict:
                model_instance = self.model_dict[model][cell]
                if model_instance.bounds is None:
                    raise ValueError(
                        "model \"{0}\" bounds not yet set".format(model)
                    )
                if model_instance.x0 is None:
                    raise ValueError(
                        "model \"{0}\" x0 not yet set".format(model)
                    )

                print("Fitting {0}".format(model))
                model_instance.fit_params(solver_params)
                # Build dict for json dump, json requires list instead of ndarray
                param_dict = {param: model_instance.fit.tolist()[index]
                              for index, param in enumerate(model_instance.param_names)}
                cell_fits[cell][model_instance.__class__.__name__] = param_dict
                cell_lls[cell][model_instance.__class__.__name__] = model_instance.fun
            print("Models fit in {0} seconds".format(time.time() - self.time_start))
            util.save_data({cell:cell_fits[cell]}, "cell_fits", path=self.save_dir, cell=cell)
            util.save_data({cell:cell_lls[cell]}, "log_likelihoods", path=self.save_dir, cell=cell)

    def _do_compare(self, model_min_name, model_max_name, cell, p_value, smoother_value):
        """Runs likelhood ratio test.

        """
        try:
            min_model = self.model_dict[model_min_name][cell]
        except:
            raise NameError(
                "Supplied model \"{0}\" has not been fit".format(model_min_name))
        try:
            max_model = self.model_dict[model_max_name][cell]
        except:
            raise NameError(
                "Supplied model \"{0}\" has not been fit".format(model_max_name))

        if len(max_model.param_names) - len(min_model.param_names) < 1:
            raise ValueError(
                "Supplied models appear to be uncomparable. min_model has same or greater # of parameters"
            )

        print("{0} fit is: {1}".format(model_min_name, min_model.fit))
        print("{0} fit is: {1}".format(model_max_name, max_model.fit))
        outcome = str(self.lr_test(
            min_model.fun,
            max_model.fun,
            p_value,
            len(max_model.param_names) - len(min_model.param_names)
        ))
        comparison_name = max_model.__class__.__name__+"_"+min_model.__class__.__name__
        util.update_comparisons(str(cell), comparison_name, outcome, path=self.save_dir)
        print(outcome)
        cellplot.plot_comparison(
            self.data_processor.spikes_summed[cell],
            min_model,
            max_model,
            cell,
            smoother_value=smoother_value)
        plt.show()

        return outcome

    def compare_models(self, model_min_name, model_max_name, p_value, smoother_value):
        """Runs likelihood ratio test on likelihoods from given nested model parameters then saves to disk.

        Parameters
        ----------
        model_min_name : string
            Model chosen for comparison with lower number of parameters.
            Name must match implementation.
        model_max_name : string
            Model chosen for comparison with greater number of parameters.
            Name must match implementation.
        p_value : float
            Threshold for likelihood ratio test such that the nested model is considered better.

        """
        outcomes = {cell: self._do_compare(
            model_min_name, model_max_name, cell, p_value, smoother_value) for cell in self.cell_range}

    def compare_even_odd(self, model_min_name, model_max_name, p_threshold):
        """Runs likelihood ratio test on even and odd trial's likelihoods from given nested model parameters then saves to disk.

        Parameters
        ----------
        model_min_name : string
            Model chosen for comparison with lower number of parameters.
            Name must match implementation.
        model_max_name : string
            Model chosen for comparison with greater number of parameters.
            Name must match implementation.
        p_value : float
            Threshold for likelihood ratio test such that the nested model is considered better.

        """
        for cell in self.cell_range:
            oddpath = os.getcwd() + "/results/log_likelihoods_odd_{0}.json".format(cell)
            evenpath = os.getcwd() + "/results/log_likelihoods_even_{0}.json".format(cell)

            if os.path.exists(oddpath) and os.path.exists(evenpath):
                with open(oddpath) as f:
                    odd_ll = json.load(f)
                with open(evenpath) as f:
                    even_ll = json.load(f)
            else:
                raise FileNotFoundError(
                    "Even or odd log likelihoods not found"
                )
                
            min_model = self.model_dict[model_min_name][cell]
            max_model = self.model_dict[model_max_name][cell]
            delta_params = len(max_model.param_names) - len(min_model.param_names)
            outcome_odd = self.lr_test(odd_ll[str(cell)][model_min_name], odd_ll[str(
                cell)][model_max_name], p_threshold, delta_params)
            outcome_even = self.lr_test(even_ll[str(cell)][model_min_name], even_ll[str(
                cell)][model_max_name], p_threshold, delta_params)
            maxname = max_model.__class__.__name__
            minname = min_model.__class__.__name__
            comparison_name = max_model.__class__.__name__+"_"+min_model.__class__.__name__
            util.update_comparisons(str(cell), comparison_name, str(outcome_odd), path=self.save_dir, odd_even="odd")
            util.update_comparisons(str(cell), comparison_name, str(outcome_even), path =self.save_dir, odd_even="even")

    def lr_test(self, ll_min, ll_max, p_threshold, delta_params):
        """Performs likelihood ratio test.

        Parameters
        ----------
        ll_min : float
            Log likelihood of model with fewer params.
        ll_max
            Log likelihood of model with more params.
        p_threshold : float
            Threshold of p value to accept model_max as better.
        delta_params : int
            Difference in number of parameters in nested model.

        Returns
        -------
        bool
            True if test passes.

        """
        lr = -2 * (ll_max - ll_min)  # log-likelihood ratio
        p = chi2.sf(lr, delta_params)
        print(ll_min, ll_max, delta_params)
        print("p-value is: " + str(p))
        return p < p_threshold

    def show_condition_fit(self, model, smoother_value):
        """Plots data and fits against provided conditions.

        Parameters
        ----------
        model : str
            Name of model

        """
        for cell in self.cell_range:
            extracted_model = self.model_dict[model][cell]

            cellplot.plot_cat_fit(
                extracted_model, cell, self.data_processor.spikes_summed_cat[cell], self.subsample, smoother_value=smoother_value)
            plt.show()

    def show_rasters(self):
        """Plots binned spike raster and saves to disk.

        """
        for cell in self.cell_range:

            cellplot.plot_raster_spiketrain(
                self.data_processor.spikes_summed[cell], self.data_processor.spikes_binned[cell], self.data_processor.window, cell)
            plt.show()