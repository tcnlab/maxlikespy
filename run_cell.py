import sys
from python_data import DataProcessor
from region_info import RegionInfo
from analysis_pipeline import AnalysisPipeline
import json
import os
import util


def run_script(cell_range):


    # path_to_data = '/Users/stevecharczynski/workspace/data/brincat_miller'
    # path_to_data = "/usr3/bustaff/scharcz/workspace/brincat_miller/"
    # time_info = RegionInfo(500, 1750, 1.0)
    # data_processor = DataProcessor(path_to_data, cell_range, 0, time_info=time_info)
    # n = 2
    # mean_delta = 0.10 * (time_info.region_high - time_info.region_low)
    # mean_bounds = (
    #     (time_info.region_low - mean_delta),
    #     (time_info.region_high + mean_delta))
    # swarm_params = {
    #                 "phip" : 0.5,
    #                 "phig" : 0.6,
    #                 "omega" : 0.6,
    #                 "minstep" : 1e-10,
    #                 "minfunc" : 1e-10,
    #                 "maxiter" : 1000
    #             }
    # bounds = ((0, 1 / n), mean_bounds, (10, 1000), (10**-10, 1 / n))
    # bounds_c = [(10**-10, 0.99)]
    # pipeline = AnalysisPipeline(cell_range, data_processor, ["Const", "Time"], 0, swarm_params)
    # pipeline.set_model_bounds("Time", bounds)
    # pipeline.set_model_bounds("Const", bounds_c)
    # pipeline.fit_all_models(1)
    # pipeline.compare_models("Const", "Time")


    path_to_data = '/Users/stevecharczynski/workspace/rui_fake_cells/mixed_firing'
    time_info = RegionInfo(0, 2000, 1.0)
    # data_descriptor = DescribeData(path_to_data, False, "ms", [44,56], 0, time_info=time_info)
    # data_descriptor = DescribeData(path_to_data, False, "ms", 60, 0)
    # cell_range = [44,45]
    data_processor = DataProcessor(path_to_data, cell_range, 0, time_info=time_info)
    n = 2
    bounds = [(0, 1 / n), (-500, 2500), (0.01, 5000), (10**-10, 1 / n)]
    bounds_c = [(10**-10, 0.99)]
    swarm_params = {
                    "phip" : 0.5,
                    "phig" : 0.5,
                    "omega" : 0.5,
                    "minstep" : 1e-10,
                    "minfunc" : 1e-10,
                    "maxiter" : 1000
                }
    pipeline = AnalysisPipeline(cell_range, data_processor, ["Const", "Time"], 0, swarm_params)
    pipeline.set_model_bounds("Time", bounds)
    pipeline.set_model_bounds("Const", bounds_c)
    pipeline.fit_all_models(1)
    pipeline.compare_models("Const", "Time")

    
    util.collect_data(cell_range, "log_likelihoods")
    util.collect_data(cell_range, "model_comparisons")
    util.collect_data(cell_range, "cell_fits")
# run_script([11,12])
if __name__ == "__main__":
    cell_range = sys.argv[-2:]
    cell_range = list(map(int, cell_range))
    cell_range = range(cell_range[0], cell_range[1]+1)
    run_script(cell_range)