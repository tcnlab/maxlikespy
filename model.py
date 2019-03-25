import numpy as np
from pyswarm import pso
from scipy.optimize import minimize
import autograd
from scipy.optimize import basinhopping


class RandomDisplacementBounds(object):
    """random displacement with bounds"""
    def __init__(self, xmin, xmax, stepsize=1000):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds"""

        min_step = np.maximum(self.xmin - x, -self.stepsize)
        max_step = np.minimum(self.xmax - x, self.stepsize)
        print("min_step {0}".format(min_step))
        print("max_step {0}".format(max_step))
        random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
        # random_step = np.random.normal((min_step+max_step)/2, self.stepsize, size=x.shape)

        print("random_step {0}".format(random_step))
        print("x is {0}".format(x))
        xnew = x + random_step
        # if x <= self.xmin:
        #     xnew = self.xmin
        # elif x >= self.xmax:
        #     xnew = self.xmax
        # else:       
        #     while True:
        #         # this could be done in a much more clever way, but it will work for example purposes
        #         xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
        #         if np.all(xnew < self.xmax) and np.all(xnew > self.xmin):
        #             break
        
        print("new step is {0}".format(xnew))

        return xnew

class MyBounds(object):
    def __init__(self, xmax, xmin):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

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
        print("are we in here?")
        self.time_info = data['time_info']
        self.t = np.arange(
            self.time_info[0],
            self.time_info[1],
            1)
        self.num_trials = data['num_trials']
        self.bounds = None
        self.x0 = None

    def fit_params(self, solver_params):
        """Fit model paramters using Particle Swarm Optimization then SciPy's minimize.

        Returns
        -------
        tuple of list and float
            Contains list of parameter fits and the final function value.

        """
        stepper = RandomDisplacementBounds(self.lb, self.ub)
        accepter = MyBounds(self.ub, self.lb)
        if solver_params["use_jac"]:
            minimizer_kwargs = {"method":solver_params["method"], "bounds":self.bounds, "jac":autograd.jacobian(self.objective)}
        else:
            minimizer_kwargs = {"method":solver_params["method"], "bounds":self.bounds}

        second_pass_res = basinhopping(
            self.objective,
            self.x0,
            disp=False,
            niter=solver_params["niter"],
            accept_test=accepter,
            take_step=stepper,  
            stepsize=solver_params["stepsize"],
            minimizer_kwargs=minimizer_kwargs,
            interval=solver_params["interval"],
        )
        
        self.fit = second_pass_res.x
        self.fun = second_pass_res.fun
        
        return (self.fit, self.fun)

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

    def pso_con(self, x):
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

    def model(self):
        """Defines functional form of model.

        Returns
        -------
        ndarray
            Model function

        """
        raise NotImplementedError("Must override model")

    def objective(self):
        """Defines objective function for minimization.

        Returns
        -------
        ndarray
            Objective function (log likelihood)

        """
        raise NotImplementedError("Must override objective")

    def set_bounds(self, bounds):
        """Set parameter bounds for solver - required.

        Parameters
        ----------
        bounds : array of tuples
            Tuples consisting of lower and upper bounds per parameter.
            These must be passed in the same order as defined in the model.

        """       
        if len(bounds) != len(self.param_names):
                raise AttributeError("Wrong number of bounds supplied")
        self.bounds = [bounds[x] for x in self.param_names]
        self.lb = [bounds[x][0] for x in self.param_names]
        self.ub = [bounds[x][1] for x in self.param_names]

    def set_info(self, name, data):
        self.info[name] = data

        return True

