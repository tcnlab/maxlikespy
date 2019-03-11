import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import os


def plot_raster(spikes, time_info, condition=0, conditions=0):
    """Plots raster of binned spike data.

    Parameters
    ----------
    spikes : ndarray
        Binned spike data for the given cell
    time_info : RegionInfo
        Object that holds timing information including the beginning and
        end of the region of interest. All in milliseconds.
    condition : int
        The condition of interest in a single condition raster, if provided.
    conditions : dict
        dict of condition data, if provided.

    """
    if condition:
        scatter_data = np.nonzero(spikes.T * conditions[condition])
    else:
        scatter_data = np.add(np.nonzero(spikes.T), time_info[0])

    plt.scatter(scatter_data[0], scatter_data[1],
                c=[[0, 0, 0]], marker="o", s=1)

def plot_spike_train(spike_train):

    plt.plot(smooth_spikes(spike_train))


def plot_cat_fit(model, cell_num, spikes, subsample=0):
    fig = plt.figure()
    num_conditions = len(model.conditions)
    fig.suptitle("cell " + str(cell_num))
    fig_name = "results/figs/cell_%d_" + model.__class__.__name__ + ".png"
    plt.legend(loc="upper left")
    window = np.arange(
            model.time_info[0],
            model.time_info[1],
            1)

    for condition in model.conditions:
        if condition:
            plt.subplot(2, num_conditions, condition + 1)
            plt.plot(window, model.expose_fit(condition), label="fit")
            plt.plot(window,
                     smooth_spikes(spikes, model.num_trials, subsample=subsample, condition=condition), label="spike_train")

    fig.savefig(fig_name % cell_num)


def plot_comparison(spikes, model_min, model_max, cell_no):
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
    window = np.arange(
        model_min.time_info[0],
        model_min.time_info[1],
        1)
    fig = plt.figure()
    fig.suptitle("cell " + str(cell_no))
    fig_name = os.getcwd() + "/results/figs/cell_%d_" + model_min.__class__.__name__ + \
        "_" + model_max.__class__.__name__ + ".png"

    plt.subplot(2, 1, 1)
    plot_fit(model_min)
    plt.plot(window, smooth_spikes(
        spikes, model_max.num_trials), label="spike_train")
    plot_fit(model_max)
    plt.legend(loc="upper right")

    plt.subplot(2, 1, 2)
    plot_raster(model_max.spikes, model_max.time_info)
    plt.xlim(model_max.time_info[0], model_max.time_info[1])
    fig.savefig(fig_name % cell_no)


def plot_fit(model):
    """Plot parameter fit over region of interest.

    Parameters
    ----------
    model : Model
        Model desired for plotting.

    """
    window = np.arange(
        model.time_info[0],
        model.time_info[1],
        1)
    fit = model.model(model.fit)
    try:
        plt.plot(window, fit, label=model.__class__.__name__)
    except:
        if fit.size == 1:
            plt.plot(window, np.full(model.t.shape, fit), label=model.__class__.__name__ )



def smooth_spikes(spikes, num_trials=0, subsample=0, condition=0):
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
    if not num_trials:
        return scipy.ndimage.filters.gaussian_filter(spikes, 100)

    if subsample:
        num_trials = int(num_trials / subsample)
    if condition:
        avg_spikes = spikes[condition] / int(num_trials)
    else:
        avg_spikes = spikes / int(num_trials)

    return scipy.ndimage.filters.gaussian_filter(avg_spikes, 100)
