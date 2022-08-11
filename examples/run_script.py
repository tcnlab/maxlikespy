import maxlikespy.analysis as analysis


path_to_data = "input_data/"
save_dir = "results/"
cell_range = range(0, 400)
data_processor = analysis.DataProcessor(path_to_data, cell_range, [0, 10000])
solver_params = {
    "niter": 25,
    "stepsize": 200,
    "interval": 10,
    "method": "TNC",
    "use_jac": True,
    "T" : 1,
    "disp":False
}
bounds_t = {
    "a_1": [10e-10, 1 / 2],
    "ut": [-1000, 12000.],
    "st": [100, 20000.],
    "a_0": [10e-10, 1 / 2]
}
pipeline = analysis.Pipeline(cell_range, data_processor, [
                            "Const", "Time"], save_dir=save_dir)
pipeline.set_model_bounds("Time", bounds_t)
pipeline.set_model_bounds("Const", {"a_0":[10**-10, 1]})
pipeline.set_model_x0("Time", [1e-5, 2000, 200, 1e-5])
pipeline.set_model_x0("Const", [1e-5])
pipeline.fit_even_odd(solver_params=solver_params)
pipeline.fit_all_models(solver_params=solver_params)
pipeline.compare_models("Const", "Time", 0.001, smoother_value=100)
pipeline.compare_even_odd("Const", "Time", 0.001)
