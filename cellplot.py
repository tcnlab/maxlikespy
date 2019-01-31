import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.signal
from models import Const
import matplotlib as mpl
import os


class CellPlot(object):

    """Provides various common plotting functions.

    Parameters
    ----------
    summed_spikes : ndarray
        Summed spike data for the given cell.

    """

    def __init__(self, summed_spikes, conditions=0):
        self.summmed_spikes = summed_spikes

    def plot_raster(self, spikes, time_info, condition=0, conditions=0):
        """Plots raster of binned spike data.
        
        Parameters
        ----------
        spikes : ndarray
            Binned spike data for the given cell
        time_info : RegionInfo
            Object that holds timing information including the beginning and
            end of the region of interest and the time bin. All in milliseconds.
        condition : int
            The condition of interest in a single condition raster, if provided.
        conditions : dict
            dict of condition data, if provided.

        """
        if condition:
            scatter_data = np.nonzero(spikes.T * conditions[condition])
        else:
            scatter_data = np.add(np.nonzero(spikes.T), time_info.region_low)

        plt.scatter(scatter_data[0], scatter_data[1], c=[[0,0,0]], marker="o", s=1)

    def plot_cat_fit(self, model):
        fig = plt.figure()    
        num_conditions = len(model.conditions)
        fig.suptitle("cell " + str(self.cell_no))
        fig_name = "figs/cell_%d_" + model.name + ".png"
        plt.legend(loc="upper left")

        for condition in model.conditions:
            plt.subplot(2, num_conditions, condition + 1)
            plt.plot(model.region, model.expose_fit(condition), label="fit")
            plt.plot(model.region, self.smooth_spikes(self.get_model_sum(model, True)[condition]), label="spike_train")

        fig.savefig(fig_name % self.cell_no)

    def plot_comparison(self, model_min, model_max, cell_no):
        """Given two models, produces a comparison figure and saves to disk.

        Parameters
        ----------
        model_min : Model
            Model chosen for comparison with lower number of parameters.
        model_max : Model
            Model chosen for comparison with greater number of parameters.
        cell_no : int
            Numeric label of cell, used for plot title.

        """
        fig = plt.figure()
        fig.suptitle("cell " + str(cell_no))
        fig_name = os.getcwd() + "/results/figs/cell_%d_" + model_min.name + "_" + model_max.name + ".png"

        plt.subplot(2,1,1)
        self.plot_fit(model_min)
        plt.plot(model_max.region, self.smooth_spikes(self.summmed_spikes, model_max.num_trials), label="spike_train")
        self.plot_fit(model_max)
        plt.legend(loc="upper right")

        plt.subplot(2,1,2)
        self.plot_raster(model_max.spikes, model_max.time_info)
        plt.xlim(model_max.time_info.region_low, model_max.time_info.region_high)
        fig.savefig(fig_name % cell_no)

    def plot_fit(self, model):
        """Plot parameter fit over region of interest.

        Parameters
        ----------
        model : Model
            Model desired for plotting.

        """
        if isinstance(model, Const):
            # plt.axhline(y=model.fit, color='r', linestyle='-')
            x1, x2 = [model.region.region_low, model.fit], [model.region.region_high, model.fit]
            plt.plot(x1, x2)
        else:
            plt.plot(model.region, model.expose_fit(), label=model.name)

    def smooth_spikes(self, spikes, num_trials, subsample=0):
        """Applys a gaussian blur filter to spike data.

        Parameters
        ----------
        spikes : ndarray
            Binned spike data.
        num_trials : int
            Number of trials given cell has in data.
        subsample : float
            Number signifying the percentage of trials to be used, 0 will use all trials.

        """
        if subsample:
            avg_spikes = spikes / int(num_trials / subsample)
        else:
            avg_spikes = spikes / int(num_trials)

        return scipy.ndimage.filters.gaussian_filter(avg_spikes, 50)

