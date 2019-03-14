import models
import time
import matplotlib.pyplot as plt
import cellplot
import numpy as np
import os
import json
import sys
from scipy.stats import chi2
import util


class AnalysisPipeline(object):

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
    swarm_params : dict of float
        Dict containing parameters for the particle swarm algorithm. Has default values.

    Attributes
    ----------
    time_start : float
        Time keeping variable for diagnostic purposes.
    cell_range : range
        Beginning and end cell to be analyzed, in range format.
    data_processor : DataProcessor
        Object returned by data processing module that includes all relevent cell data.
    time_info : TimeInfo
        Object that holds timing information including the beginning and end of the region
        of interest and the time bin. All in seconds.
    analysis_dict : dict (int: AnalyzeCell)
        Dict containing model fits per cell as contained in AnalyzeCell objects.
    model_fits : dict (str: dict (int: Model))
        Nested dictionary containing all model fits, per model per cell.
    subsample : float
        Number signifying the percentage of trials to be used, 0 will use all trials.
    model_dict : dict of Model
        Dict that contains all initialized Model instances.

    """

    def __init__(self, cell_range, data_processor, models, subsample=0, swarm_params=None):
        self.time_start = time.time()
        self.cell_range = cell_range
        self.data_processor = data_processor
        self.subsample = subsample
        if not swarm_params:
            self.swarm_params = {
                "phip": 0.5,
                "phig": 0.5,
                "omega": 0.5,
                "minstep": 1e-8,
                "minfunc": 1e-8,
                "maxiter": 1000
            }
        else:
            self.swarm_params = swarm_params
        self.model_dict = self._make_models(models)

    def _make_models(self, models_to_fit):
        """Initializes and creates dict of models to fit.

        """
        model_dict = {model: {} for model in models_to_fit}
        condition_data = self.data_processor.conditions_dict
        spike_info = self.data_processor.spike_info
        for cell in self.cell_range:
            if self.subsample:
                sampled_trials = self._subsample_trials(
                    self.data_processor.num_trials[cell], self.subsample)
                num_trials = len(sampled_trials)
                spikes_binned = self._apply_subsample(
                    self.data_processor.spikes_binned[cell],
                    sampled_trials)
                if condition_data:
                    conditions = {cond: condition_data[cell][cond][sampled_trials]
                                  for cond in condition_data[cell]}
            else:
                num_trials = self.data_processor.num_trials[cell]
                spikes_binned = self.data_processor.spikes_binned[cell]
                if condition_data:
                    conditions = condition_data[cell]

            for model in models_to_fit:

                # data passed here is determined by what models need
                model_data = {}
                model_data['spikes'] = spikes_binned
                model_data['time_info'] = self.data_processor.time_info
                model_data['num_trials'] = num_trials
                if condition_data:
                    model_data['conditions'] = conditions
                if spike_info:
                    model_data['spike_info'] = spike_info[str(cell)]
                model_data['swarm_params'] = self.swarm_params
                # this creates an instance of class "model" in the module "models"
                try:
                    model_instance = getattr(models, model)(model_data)
                    model_dict[model][cell] = model_instance
                except:
                    raise NameError(
                        "Supplied model \"{0}\" does not exist".format(model))

        return model_dict

    def _apply_subsample(self, spikes, sampled_trials):
        """Returns spikes for the chosen subset of trials"

        """

        return spikes[sampled_trials, :]

    @staticmethod
    def _subsample_trials(num_trials, subsample):
        """Randomly selects a percentage of total trials

        """
        num_trials = int(num_trials * subsample)
        if num_trials < 1:
            num_trials = 1
        sampled_trials = np.random.randint(
            num_trials,
            size=num_trials)

        return sampled_trials

    def set_model_bounds(self, model, bounds):
        """Sets solver bounds for the given model for all cells.

        Parameters
        ----------
        model : string
            String with same name as desired model.

        bounds : list of tuples of float
            List of lower and upper bounds for the solver.

        """
        for cell in self.cell_range:
            if model in self.model_dict:
                self.model_dict[model][cell].set_bounds(bounds)
            else:
                raise ValueError("model does not match supplied models")

        return True

    def set_model_info(self, model, name, data):
        for cell in self.cell_range:
            if model in self.model_dict:
                self.model_dict[model][cell].set_info(name, data)
            else:
                raise ValueError("model does not match supplied models")

        return True

    def fit_all_models(self, iterations):
        """Fits parameters for all models then saves to disk.

        Parameters
        ----------
        iterations : int
            The number of iterations the solver must reach without likelihood improvement
            before terminating. 

        """
        cell_fits = {}
        cell_lls = {}
        for cell in self.cell_range:
            print(cell)
            cell_fits[cell] = {}
            cell_lls[cell] = {}
            for model in self.model_dict:
                model_instance = self.model_dict[model][cell]
                if model_instance.bounds is None:
                    raise ValueError("model \"{0}\" bounds not yet set".format(model))
                print("fitting {0}".format(model))
                self._fit_model(model_instance, iterations)
                param_dict = {param: model_instance.fit.tolist()[index] 
                    for index, param in enumerate(model_instance.param_names)}
                cell_fits[cell][model_instance.__class__.__name__] = param_dict
                cell_lls[cell][model_instance.__class__.__name__] = model_instance.fun

            util.save_data(cell_fits, "cell_fits", cell=cell)
            util.save_data(cell_lls, "log_likelihoods", cell=cell)

        return True

    def _fit_model(self, model, iterations):
        """Fit given model parameters.

        """
        self._iterate_fits(model, iterations)
        return model

    @staticmethod
    def _iterate_fits(model, n):
        """Performs fitting, checking if fit is improved for n iterations.

        """
        iteration = 0
        fun_min = sys.float_info.max
        while iteration < n:
            model.fit_params()
            # check if the returned fit is better by at least a tiny amount
            if model.fun < (fun_min - fun_min * 0.0001):
                fun_min = model.fun
                params_min = model.fit
                iteration = 0
            else:
                iteration += 1
        model.fit = params_min
        model.fun = fun_min
    
        return model

    def _do_compare(self, model_min, model_max, cell, p_value):
        """Internally runs likelhood ratio test.

        """

        # possibly rewrite to create one CellPlot and pass params for plotting
        try:
            min_model = self.model_dict[model_min][cell]
        except:
            raise NameError(
                "Supplied model \"{0}\" has not been fit".format(model_min))
        try:
            max_model = self.model_dict[model_max][cell]
        except:
            raise NameError(
                "Supplied model \"{0}\" has not been fit".format(model_max))

        print(min_model.fit)
        print(max_model.fit)
        outcome = str(self.lr_test(
            min_model,
            max_model,
            p_value
        ))
        outcome_dict = {cell:
                        {max_model.__class__.__name__+"_"+min_model.__class__.__name__: outcome}}
        util.save_data(data=outcome_dict,
                            filename="model_comparisons",
                            cell=cell)
        print(outcome)
        cellplot.plot_comparison(
            self.data_processor.spikes_summed[cell],
            min_model,
            max_model,
            cell)
        print("TIME IS")
        print(time.time() - self.time_start)
        plt.show()

        return outcome

    def compare_models(self, model_min, model_max, p_value):
        """Runs likelihood ratio test on likelihoods from given nested model parameters then saves to disk.

        Parameters
        ----------
        model_min : string
            Model chosen for comparison with lower number of parameters.
            Name must match implementation.
        model_max : string
            Model chosen for comparison with greater number of parameters.
            Name must match implementation.

        """
        outcomes = {cell: self._do_compare(
            model_min, model_max, cell, p_value) for cell in self.cell_range}

        return True

    def lr_test(self, model_min, model_max, p_threshold):
        """Performs likelihood ratio test.

        Parameters
        ----------
        model_min : Model
            Model chosen for comparison with lower number of parameters.
        model_max : Model
            Model chosen for comparison with greater number of parameters.
        p_threshold : float
            Threshold of p value to accept model_max as better.

        Returns
        -------
        Boolean
            True if test passes.

        """
        llmin = model_min.fun
        llmax = model_max.fun
        delta_params = len(model_max.param_names) - len(model_min.param_names)
        lr = -2 * (llmax - llmin)  # log-likelihood ratio
        p = chi2.sf(lr, delta_params)
        print(llmin, llmax, delta_params)
        print("p-value is: " + str(p))
        return p < p_threshold

    def show_condition_fit(self, model):
        for cell in self.cell_range:
            extracted_model = self.model_dict[model][cell]

            cellplot.plot_cat_fit(
                extracted_model, cell, self.data_processor.spikes_summed_cat[cell], self.subsample)
            plt.show()
