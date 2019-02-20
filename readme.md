# Maximum Likelihood in Python

This project is a translation of Matlab code that accepts/rejects maximum likelihood models of neural spike trains.

## Note
This project is a WIP.


## Getting Started

Clone repo locally.

### Prerequisites
```
numpy 
scipy
pyswarm

```
## Data Formatting

Spike data currently is expected to be in json formatted arrays of dimension NumberOfCells X MaxTrials, with spikes in milliseconds. Data processing code expects one file per cell, in a subdirectory named "spikes".

In addition to spikes, "number_of_trials.json" must be created to denote how many trials exist for each cell.

Supplentary information supported currently:
  
  A file "conditions.json" including integer labels per cell and trial.
  
  A file "spike_info.json", a json-serialized dict containing labeled information on a millisecond resolution (such as animal position).
  
Examples for these files will be provided in /examples.

## Usage

In /examples there is an example script going through a full analysis routine with a sample model, however some basic usage is outlined below.

An instance of two required classes must be initialized:

  `DataProcessor(path, cell_range, time_info=None)` where
  * path - absolute file path to data folder
  * cell_range - range indicating the first and last cell to be analyzed
  * time_info - An instance of a helper class describing the experimental time range initialized as `RegionInfo(low_bound, high_bound)`
  
 `AnalysisPipeline(cell_range, data_processor, models, subsample, swarm_params=None)`where
 * cell_range - as above
 * data_processor - the previously initialized DataProcessor object
 * models - list of models to be fit, as strings.
 * subsample - float signifying percentage of trials to be used, 0 if all are desired
 * swarm params - optional dict the user can provide if overriding the particle swarm solver's parameters is desired. this takes the form: 
 ```
 "phip" : 0.5,"phig" : 0.5, "omega" : 0.5,"minstep" : 1e-8, "minfunc" : 1e-8, "maxiter" : 1000
 ```
 It is recommended by the `pyswarm` documentation to set the first three parameters between 0 and 1. Further documentation can be found on their website.
 
 Typical usage from this stage, for example to fit and compare two nested models called "Time" and "Const", would be to first set your model upper and lower bounds. Bounds are given in the form of a list of tuples for each parameter.
 
 ```
 pipeline.set_model_bounds("Time", bounds_t)
 pipeline.set_model_bounds("Const", bounds_c)
 ```
 
 Then fit all models, where number of iterations is the number of times the solver should run without finding a better parameter fit.
 
 `pipeline.fit_all_models(number_of_iterations)`
 
Finally running

`pipeline.compare_models(min_model="Const", max_model="Time", p_value=0.01)`

will perform a likelhihood ratio test on the supplied models.

These functions will save off in /results a plot of model comparisons and json files containing the outcome of the likelihood ratio test, parameter fits, and likelhihoods per cell. 

## Adding custom models

New models should be added to models.py and inherit the Model base class as well as implement all parent methods. A model, "Time", consisting of a guassian firing field in addition to a constant firing field is provided for example.


## Built With

* [pyswarm](https://pythonhosted.org/pyswarm/) - Particle swarm solver used


## Authors

* **Stephen Charczynski** - *Initial work* - [scharczynski](https://github.com/scharczynski)


## Acknowledgments

* Zoran Tiganj [zorant](https://github.com/zorant) for writing the initial incarnation of this software.
