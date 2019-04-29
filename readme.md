# Maximum Likelihood in Python

This project is a translation of Matlab code that fits and accepts/rejects maximum likelihood models of neural spike trains.
Parameter fitting is performed using SciPy's implementation of basinhopping - a global optimization algorithm.

## Getting Started

pip install maxlikespy

### Prerequisites
```
numpy 
scipy
autograd

```
## Data Formatting

Spike data currently is expected to be in json formatted arrays of dimension NumberOfCells X MaxTrials, with spikes in milliseconds. Data processing code expects one file per cell, in a subdirectory named "spikes".

In addition to spikes, "number_of_trials.json" must be created to denote how many trials exist for each cell.

Supplentary information supported currently:
  
  "trial_lengths.json" - per trial trial lengths in case trial lengths are not fixed.
  
  "conditions.json" - integer labels per cell and trial.
  
  "spike_info.json" - json-serialized dict containing labeled information on a millisecond resolution (such as animal position).
  
Examples for these files will be provided in /examples.

## Usage

In /examples there is an example script going through a full analysis routine with a sample model, however some basic usage is outlined below.

An instance of two required classes must be initialized:

  `DataProcessor(path, cell_range, time_info=None)` where
  * path - absolute file path to data folder
  * cell_range - range indicating the first and last cell to be analyzed
  * time_info - An array describing the experimental time range `[0, 1000]`
  
 `AnalysisPipeline(cell_range, data_processor, models, subsample)`where
 * cell_range - as above
 * data_processor - the previously initialized DataProcessor object
 * models - list of models to be fit, as strings.
 * subsample - float signifying percentage of trials to be used, 0 if all are desired

 Typical usage from this stage, for example to fit and compare two nested models called "Time" and "Const", would be to first set your model upper and lower bounds. Bounds are given in the form of a list of tuples for each parameter.
 
 ```
 pipeline.set_model_bounds("Time", bounds_t)
 pipeline.set_model_bounds("Const", bounds_c)
 ```
 
 Then fit all models, where number of iterations is the number of times the solver should run without finding a better parameter fit.
 
 `pipeline.fit_all_models(solver_params)`

solver_params is a dict containing various solver parameters:
```
solver_params = {
        "niter": 100,
        "stepsize": 1000,
        "interval": 10,
        "method": "TNC",
        "use_jac": True,
    }
 ```
These params are as described in the SciPy documentation [https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html].

If `use_jac` is `True`, Autograd [https://github.com/HIPS/autograd] will be used to provide the jacobian of your model's objective function to the solver. Autograd requires variety of constraints on your objective function to work and their documentation must be read.

Finally running

`pipeline.compare_models(min_model="Const", max_model="Time", p_value=0.01)`

will perform a likelihood ratio test on the supplied models.

These functions will save off in /results a plot of model comparisons and json files containing the outcome of the likelihood ratio test, parameter fits, and likelihoods per cell. 

## Adding custom models

New models should be added to models.py and inherit the Model base class as well as implement all parent methods. A model, "Gaussian", consisting of a guassian firing field in addition to a constant firing field is provided for example.


## Built With

* [autograd](https://github.com/HIPS/autograd) - Automatic differentiation package for improved solver performance.


## Authors

* **Stephen Charczynski** - *Initial work* - [scharczynski](https://github.com/scharczynski)


## Acknowledgments

* Zoran Tiganj [zorant](https://github.com/zorant) for writing the initial incarnation of this software.
