import numpy as np
from pyswarm import pso
from scipy.optimize import minimize
from model import Model


class Time(Model):

    """Model which contains a time dependent gaussian compenent and an offset parameter.

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
        self.name = "time"
        self.region = self.t

    def build_function(self, x):
        #pso stores params in vector x
        a, ut, st, o = x[0], x[1], x[2], x[3]
 
        self.function = (
            (a * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + o)
        res = np.sum(self.spikes * (-np.log(self.function)) +
                     (1 - self.spikes) * (-np.log(1 - (self.function))))
        return res

    def fit_params(self):
        super().fit_params()
        return (self.fit, self.fun)

    def pso_con(self, x):
        return 1 - (x[0] + x[3])

    def expose_fit(self):
        if self.fit is None:
            raise ValueError("fit not yet computed")
        else:
            self.a = self.fit[0]
            self.ut = self.fit[1]
            self.st = self.fit[2]
            self.o = self.fit[3]
        fun = (self.a * np.exp(-np.power(self.t - self.ut, 2.) /
                               (2 * np.power(self.st, 2.)))) + self.o
        return fun