import numpy as np
from pyswarm import pso
from scipy.optimize import minimize
from model import Model


class Time(Model):

    """Model which contains a time dependent gaussian component and an offset parameter.

    Attributes
    ----------
    name : string
        Human readable string describing the model.
    num_params : int
        Integer signifying the number of model parameters.
    spikes : dict     
        Dict containing binned spike data for current cell.
    region : ndarray
        Copy of "t" array used in plotting. May be unneeded.

    """

    def __init__(self, data):
        super().__init__(data)
        self.spikes = data['spikes']
        self.region = self.t
        self.param_names = ["a_1", "ut", "st", "a_0"]

    def objective(self, x):
        fun = self.model(x)
        return np.sum(self.spikes * (-np.log(fun)) +
                     (1 - self.spikes) * (-np.log(1 - (fun))))

    def model(self, x):
        a, ut, st, o = x[0], x[1], x[2], x[3]

        self.function = (
            (a * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + o)
        return self.function

    def fit_params(self):
        super().fit_params()
        return (self.fit, self.fun)

    def pso_con(self, x):
        return 1 - (x[0] + x[3])