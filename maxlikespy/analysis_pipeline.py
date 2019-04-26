import maxlikespy.cellplot as cellplot
import maxlikespy.util as util
import maxlikespy.models as models
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import sys
from scipy.stats import chi2


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

    def __init__(self, cell_range, data_processor, models, subsample=0):
        self.time_start = time.time()
        self.cell_range = cell_range
        self.data_processor = data_processor
        self.subsample = subsample
        self.model_dict = self._make_models(models)

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
                model_data['time_info'] = self.data_processor.time_info
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
                if model in self.model_dict:
                    self.model_dict[model][cell].set_x0(x0)
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
                        "model \"{0}\" bounds not yet set".format(model))

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

            util.save_data({cell:fits_even[cell]}, "cell_fits_even", cell=cell)
            util.save_data({cell:lls_even[cell]}, "log_likelihoods_even", cell=cell)

            util.save_data({cell:fits_odd[cell]}, "cell_fits_odd", cell=cell)
            util.save_data({cell:lls_odd[cell]}, "log_likelihoods_odd", cell=cell)

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
                "use_jac": True,
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
                        "model \"{0}\" bounds not yet set".format(model))

                print("Fitting {0}".format(model))
                model_instance.fit_params(solver_params)
                # Build dict for json dump, json requires list instead of ndarray
                param_dict = {param: model_instance.fit.tolist()[index]
                              for index, param in enumerate(model_instance.param_names)}
                cell_fits[cell][model_instance.__class__.__name__] = param_dict
                cell_lls[cell][model_instance.__class__.__name__] = model_instance.fun

            util.save_data({cell:cell_fits[cell]}, "cell_fits", cell=cell)
            util.save_data({cell:cell_lls[cell]}, "log_likelihoods", cell=cell)

    def _do_compare(self, model_min_name, model_max_name, cell, p_value):
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
        util.update_comparisons(str(cell), comparison_name, outcome)
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

    def compare_models(self, model_min_name, model_max_name, p_value):
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
            model_min_name, model_max_name, cell, p_value) for cell in self.cell_range}

        return True

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
            odd_dict = {cell:{maxname+"_"+minname: str(outcome_odd)}}
            even_dict = {cell:{maxname+"_"+minname: str(outcome_even)}}
            comparison_name = max_model.__class__.__name__+"_"+min_model.__class__.__name__
            util.update_comparisons(str(cell), comparison_name, odd_dict)
            util.update_comparisons(str(cell), comparison_name, even_dict)

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

    def show_condition_fit(self, model):
        """Plots data and fits against provided conditions.

        Parameters
        ----------
        model : str
            Name of model

        """
        for cell in self.cell_range:
            extracted_model = self.model_dict[model][cell]

            cellplot.plot_cat_fit(
                extracted_model, cell, self.data_processor.spikes_summed_cat[cell], self.subsample)
            plt.show()

    def show_rasters(self):
        """Plots binned spike raster and saves to disk.

        """
        for cell in self.cell_range:

            cellplot.plot_raster_spiketrain(
                self.data_processor.spikes_summed[cell], self.data_processor.spikes_binned[cell], self.data_processor.time_info, cell)
            plt.show()