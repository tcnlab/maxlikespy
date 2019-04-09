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

    def __init__(self, cell_range, data_processor, models, subsample=0, split_trials=False):
        self.time_start = time.time()
        self.cell_range = cell_range
        self.data_processor = data_processor
        self.subsample = subsample
        if split_trials:
            self.model_dict_even = self._make_models(models, "even")
        self.model_dict = self._make_models(models)

    def _make_models(self, models_to_fit, even_odd=None):
        """Initializes and creates dict of models to fit.

        """
        model_dict = {model: {} for model in models_to_fit}
        condition_data = self.data_processor.conditions_dict
        spike_info = self.data_processor.spike_info
        for cell in self.cell_range:
            num_trials, spikes_binned, conditions = self._select_model_data(cell)
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
                # try:                
                   
                # except:
                #     raise NameError(
                #         "Supplied model \"{0}\" does not exist".format(model))

        return model_dict

    def _select_model_data(self, cell):
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

    def set_model_info(self, model, name, data, per_cell=False):
        for cell in self.cell_range:
            if model in self.model_dict:
                if per_cell:
                    self.model_dict[model][cell].set_info(name, data[cell])
                else:
                    self.model_dict[model][cell].set_info(name, data)
            else:
                raise ValueError("model does not match supplied models")

        return True

    def _get_even_odd_trials(self, cell, even):
        if even:
            return self.data_processor.spikes_binned[cell][::2]
        else:
            return self.data_processor.spikes_binned[cell][1::2]

    def fit_even_odd(self, solver_params=None):
        fits_even, fits_odd = {}, {}
        lls_even, lls_odd = {}, {}
        for cell in self.cell_range:
            print(cell)
            fits_even[cell] = {}
            fits_odd[cell] = {}
            lls_even[cell], lls_odd[cell] = {}, {}
            for model in self.model_dict:
                model_instance = self.model_dict[model][cell]
                if model_instance.bounds is None:
                    raise ValueError("model \"{0}\" bounds not yet set".format(model))

                #Even trials
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


            util.save_data(fits_even, "cell_fits_even", cell=cell)
            util.save_data(lls_even, "log_likelihoods_even", cell=cell)

            util.save_data(fits_odd, "cell_fits_odd", cell=cell)
            util.save_data(lls_odd, "log_likelihoods_odd", cell=cell)     

    def fit_all_models(self, solver_params=None):
        """Fits parameters for all models then saves to disk.

        Parameters
        ----------
        iterations : int
            The number of iterations the solver must reach without likelihood improvement
            before terminating. 

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
            print(cell)
            cell_fits[cell] = {}
            cell_lls[cell] = {}
            for model in self.model_dict:
                model_instance = self.model_dict[model][cell]
                if model_instance.bounds is None:
                    raise ValueError("model \"{0}\" bounds not yet set".format(model))

                print("fitting {0}".format(model))
                model_instance.fit_params(solver_params)
                # Build dict for json dump, json requires list instead of ndarray
                param_dict = {param: model_instance.fit.tolist()[index] 
                    for index, param in enumerate(model_instance.param_names)}
                cell_fits[cell][model_instance.__class__.__name__] = param_dict
                cell_lls[cell][model_instance.__class__.__name__] = model_instance.fun

            util.save_data(cell_fits, "cell_fits", cell=cell)
            util.save_data(cell_lls, "log_likelihoods", cell=cell)

        return True

    def _do_compare(self, model_min, model_max, cell, p_value):
        """Runs likelhood ratio test.

        """
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

        if len(max_model.param_names) - len(min_model.param_names) < 1:
            raise ValueError(
                "Supplied models appear to be uncomparable. min_model has same or greater # of parameters"
            ) 

        print(min_model.fit)
        print(max_model.fit)
        outcome = str(self.lr_test(
            min_model.fun,
            max_model.fun,
            p_value,
            len(max_model.param_names) - len(min_model.param_names)
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

    def compare_even_odd(self, model_min, model_max, p_threshold):
        for cell in self.cell_range:
            oddpath = os.getcwd()+"/results/log_likelihoods_odd_{0}.json".format(cell)
            evenpath = os.getcwd()+"/results/log_likelihoods_even_{0}.json".format(cell)

            if os.path.exists(oddpath) and os.path.exists(evenpath):
                with open(oddpath) as f:
                    odd_ll = json.load(f)
                with open(evenpath) as f:
                    even_ll = json.load(f)
            else:
                raise FileNotFoundError(
                    "Even or odd log likelihoods not found"
                ) 
            min_model = self.model_dict[model_min][cell]
            max_model = self.model_dict[model_max][cell]
            delta_params = len(max_model.param_names) - len(min_model.param_names)
            outcome_odd = self.lr_test(odd_ll[str(cell)][model_min], odd_ll[str(cell)][model_max], p_threshold, delta_params)
            outcome_even = self.lr_test(even_ll[str(cell)][model_min], even_ll[str(cell)][model_max], p_threshold, delta_params)
            odd_dict = {cell:
                {max_model.__class__.__name__+"_"+min_model.__class__.__name__: str(outcome_odd)}}
            even_dict = {cell:
                {max_model.__class__.__name__+"_"+min_model.__class__.__name__: str(outcome_even)}}
            util.save_data(data=odd_dict,
                    filename="model_comparisons_odd",
                    cell=cell)
            util.save_data(data=even_dict,
                    filename="model_comparisons_even",
                    cell=cell)


    def lr_test(self, ll_min, ll_max, p_threshold, delta_params):
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
        lr = -2 * (ll_max - ll_min)  # log-likelihood ratio
        p = chi2.sf(lr, delta_params)
        print(ll_min, ll_max, delta_params)
        print("p-value is: " + str(p))
        return p < p_threshold

    def show_condition_fit(self, model):
        for cell in self.cell_range:
            extracted_model = self.model_dict[model][cell]

            cellplot.plot_cat_fit(
                extracted_model, cell, self.data_processor.spikes_summed_cat[cell], self.subsample)
            plt.show()

    def show_rasters(self, save=False):
        for cell in self.cell_range:

            cellplot.plot_raster_spiketrain(self.data_processor.spikes_summed[cell], self.data_processor.spikes_binned[cell], self.data_processor.time_info, cell)
            plt.show()
