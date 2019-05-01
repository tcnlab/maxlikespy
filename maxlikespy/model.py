import numpy as np
from scipy.optimize import minimize
import autograd
from scipy.optimize import basinhopping


class RandomDisplacementBounds(object):
    
    """Custom take_step class.

    Takes random steps but guarantees step lies within the bounds.

    """

    def __init__(self, xmin, xmax, stepsize=1000):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        min_step = np.maximum(self.xmin - x, -self.stepsize)
        max_step = np.minimum(self.xmax - x, self.stepsize)
        random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
        xnew = x + random_step

        return xnew

class AccepterBounds(object):

    """Custom accept_test class.

    Accepts step if it is within supplied bounds.

    """

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
    spikes : ndarray
        Array of binary spike train data, of dimension (trials × time).
    window : ndarray
        Array containing time information. Either two elements or of size proportional to number of trials.
    t : numpy.ndarray
        Array of 1ms timeslices of size specified by window.
    fit : list
        List of parameter fits after fitting process has been completed, initially None.
    fun : float
        Value of model objective function at computed fit parameters.
    lb : list
        List of parameter lower bounds.
    ub : list
        List of parameter upper bounds.
    x0 : list
        List of parameter initial states.
    info : dict
        Dict containing user supplied supplementary model info.
    num_trials : int
        Number of trials this cell has in data.
    param_names : list of str
        List of names for model parameters. Used in setting bounds and output.

    """

    def __init__(self, data):
        self.window = data['window']
        min_time = min(self.window[:,0])
        max_time = max(self.window[:,1])
        self.spikes = data["spikes"]
        self.t = np.arange(min_time, max_time, 1)
        self.num_trials = data['num_trials']
        self.bounds = None
        self.x0 = None
        self.fit, self.fun = None, None
        self.info = {}
        self.param_names = None

    def fit_params(self, solver_params):
        """Fit model paramters using Particle Swarm Optimization then SciPy's minimize.

        Parameters
        ----------
        solver_params : dict
            Dict of parameters for minimizer algorithm.

        Returns
        -------
        tuple of list and float
            Contains list of parameter fits and the final function value.

        """
        # validates solver_params such that it includes all options 
        param_keys = ["use_jac", "method", "niter", "stepsize", "interval"]
        for key in param_keys:
            if key not in solver_params:
                raise KeyError("Solver option {0} not set".format(key))

        stepper = RandomDisplacementBounds(self.lb, self.ub, stepsize=solver_params["stepsize"])
        accepter = AccepterBounds(self.ub, self.lb)
        if solver_params["use_jac"]:
            minimizer_kwargs = {"method":solver_params["method"], "bounds":self.bounds, "jac":autograd.jacobian(self.objective)}
        else:
            minimizer_kwargs = {"method":solver_params["method"], "bounds":self.bounds, "jac":False, "options":{"disp":True}}

        second_pass_res = basinhopping(
            self.objective,
            self.x0,
            disp=True,
            T=1,
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

    def model(self, x, plot=False):
        """Defines functional form of model.
        
        Parameters
        ----------
        x : ndarray
            Array consisting of current parameter values.
        plot : bool
            Flag for plotting a visually representative version of model.

        Returns
        -------
        ndarray
            Model function

        """
        raise NotImplementedError("Must override model")

    def objective(self, x):
        """Defines objective function for minimization.

        Parameters
        ----------
        x : ndarray
            Array consisting of current parameter values.

        Returns
        -------
        ndarray
            Objective function (log likelihood)

        """
        raise NotImplementedError("Must override objective")

    def set_bounds(self, bounds):
        """Set parameter bounds for solver.
        Parameters
        ----------
        bounds : dict of two element lists
            Dict consisting of lower and upper bounds per parameter.

        """       
        if len(bounds) != len(self.param_names):
                raise AttributeError("Wrong number of bounds supplied")
        self.bounds = [bounds[x] for x in self.param_names]
        self.lb = [bounds[x][0] for x in self.param_names]
        self.ub = [bounds[x][1] for x in self.param_names]

    def set_x0(self, x0):
        """Set initial state for model parameters.

        Parameters
        ----------
        x0 : list of float
            List consisting of x0 values to be set.

        """
        if len(x0) != len(self.param_names):
            raise AttributeError("Wrong number of initial parameter values supplied")
        self.x0 =  x0

    def set_info(self, name, data):
        """Provides ability to give arbitrary hashable data to models.
        
        Parameters
        ----------
        name : str
            Name of provided data.
        data : hashable
            Data to be given to model.

        """
        self.info[name] = data
        if "info_callback" in dir(self):
            self.info_callback()

