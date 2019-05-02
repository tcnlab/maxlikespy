import maxlikespy.analysis as analysis


path_to_data = '/Users/stevecharczynski/workspace/data/jay/2nd_file'
cell_range = range(5,10)
data_processor = analysis.DataProcessor(path_to_data, cell_range)
solver_params = {
    "niter": 2,
    "stepsize": 10000,
    "interval": 10,
    "method": "TNC",
    "use_jac": True,
    "T": 1, 
    "disp" : False
}
bounds_t = {
    "a_1": [0, 1 / 2],
    "ut": [-1000,10000],
    "st": [100, 10000],
    "a_0": [10**-10, 1 / 2]
}
bounds_c = {
    "a_0": [10**-10, 1 / 2]
}
pipeline = analysis.Pipeline(cell_range, data_processor, [
                            "Const", "Time"], 0)
pipeline.set_model_bounds("Time", bounds_t)
pipeline.set_model_bounds("Const", bounds_c)
pipeline.set_model_x0(["Const", "Time"], [[1e-5], [1e-5, 100, 100, 1e-5]])
pipeline.fit_all_models(solver_params=solver_params)
pipeline.compare_models("Const", "Time", 0.01)

