import sys
import maxlikespy.data_processor as dp
import maxlikespy.analysis_pipeline as ap
import json
import os
import maxlikespy.util as util
import numpy as np


def run_script(cell_range):

    path_to_data = '/Users/stevecharczynski/workspace/data/jay/2nd_file'
    data_processor = dp.DataProcessor(path_to_data, cell_range)
    solver_params = {
        "niter": 2,
        "stepsize": 10000,
        "interval": 10,
        "method": "TNC",
        "use_jac": True,
    }
    bounds_t = {
        "a_1": [0, 1 / n],
        "ut": [-1000,10000],
        "st": [100, 10000],
        "a_0": [10**-10, 1 / n]
    }
    bounds_c = {
        "a_0": [10**-10, 1 / n]
    }
    pipeline = ap.AnalysisPipeline(cell_range, data_processor, [
                                "Const", "Time"], 0)
    pipeline.set_model_bounds("Time", bounds_t)
    pipeline.set_model_bounds("Const", bounds_c)
    pipeline.set_model_x0(["Const", "Time"], [[1e-5], [1e-5, 100, 100, 1e-5]])
    pipeline.fit_even_odd(solver_params)
    pipeline.fit_all_models(solver_params=solver_params)
    pipeline.compare_models("Const", "Time", 0.01)


if __name__ == "__main__":
    cell_range = sys.argv[-2:]
    cell_range = list(map(int, cell_range))
    cell_range = range(cell_range[0], cell_range[1]+1)
    run_script(cell_range)
