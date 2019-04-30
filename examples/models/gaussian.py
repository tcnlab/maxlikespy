import numpy as np
from maxlikespy.model import Model
import autograd.numpy as np


class Gaussian(Model):

    """Model which contains a time dependent gaussian compenent 
    and an offset parameter.

    Attributes
    ----------
    name : string
        Human readable string describing the model.
    num_params : int
        Integer signifying the number of model parameters.
    spikes : dict     
        Dict containing binned spike data for current cell.

    """

    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["a_1", "ut", "st", "a_0"]

    def objective(self, x):
        fun = self.model(x)
        obj = np.sum(self.spikes * (-np.log(fun)) +
                      (1 - self.spikes) * (-np.log(1 - (fun))))
        
        return obj

    def model(self, x, plot=False):
        a, ut, st, o = x

        self.function = (
            (a * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + o)
            
        return self.function
