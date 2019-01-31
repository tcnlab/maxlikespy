import numpy as np
from pyswarm import pso
from scipy.optimize import minimize


class Model(object):

    """Base class for models.

    Provides common class attributes.

    Parameters
    ----------
    data : dict
        Dict containing all needed input data.

    Attributes
    ----------
    spikes : numpy.ndarray
        Array of binary spike train data, of dimension (trials × time).
    time_info : TimeInfo
        Object that holds timing information including the beginning and end of the region
        of interest and the time bin. All in seconds.
    t : numpy.ndarray
        Array of timeslices of size specified by time_low, time_high and time_bin.
    fit : list
        List of parameter fits after fitting process has been completed, initially None.
    fun : float
        Value of model objective function at computed fit parameters.
    lb : list
        List of parameter lower bounds.
    ub : list
        List of parameter upper bounds.

    """

    def __init__(self, data):
        self.time_info = data['time_info']
        self.t = np.arange(
            self.time_info.region_low,
            self.time_info.region_high,
            self.time_info.region_bin)
        self.num_trials = data['num_trials']
        self.swarm_params = data['swarm_params']

    def fit_params(self):
        """Fit model paramters using Particle Swarm Optimization then SciPy's minimize.

        Returns
        -------
        tuple of list and float
            Contains list of parameter fits and the final function value.

        """
        fit_pso, fun_pso = pso(
            self.build_function,
            self.lb, 
            self.ub,
            phip=self.swarm_params["phip"],
            phig=self.swarm_params["phig"],
            omega=self.swarm_params["omega"],
            minstep=self.swarm_params["minstep"],
            minfunc=self.swarm_params["minfunc"],
            maxiter=self.swarm_params["maxiter"], #800 is arbitrary, doesn't seem to get reached
            f_ieqcons=self.pso_con
        )
        second_pass_res = minimize(
            self.build_function,
            fit_pso,
            method='L-BFGS-B',
            bounds=self.bounds,
            options={'disp': False}
        )
        self.fit = second_pass_res.x
        self.fun = second_pass_res.fun
        return (self.fit, self.fun)
        # self.fit = fit_pso
        # self.fun = fun_pso
        # return(self.fit, self.fun)

    def build_function(self):
        """Embed model parameters in model function.

        Parameters
        ----------
        x : numpy.ndarray
            Contains optimization parameters at intermediate steps.

        Returns
        -------
        float
            Negative log-likelihood of given model under current parameter choices.

        """
        raise NotImplementedError("Must override build_function")

    def pso_con(self):
        """Define constraint on coefficients for PSO

        Note
        ----
        Constraints for pyswarm module take the form of an array of equations
        that sum to zero.

        Parameters
        ----------
        x : numpy.ndarray
            Contains optimization parameters at intermediate steps.

        """
        raise NotImplementedError("Must override pso_con")

    def expose_fit(self):
        """Returns instance of model function for plotting.

        Returns
        -------
        ndarray
            Representation of function value over interval with last fit values.

        """
        raise NotImplementedError("Must override plot_fit")

    def set_bounds(self, bounds):
        """Set parameter bounds for solver - required.

        Parameters
        ----------
        bounds : array of tuples
            Tuples consisting of lower and upper bounds per parameter.
            These must be passed in the same order as defined in the model.

        """
        self.bounds = bounds
        self.lb = [x[0] for x in self.bounds]
        self.ub = [x[1] for x in self.bounds]

