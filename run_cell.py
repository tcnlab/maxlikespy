import sys
from python_data import DataProcessor
from region_info import RegionInfo
from analysis_pipeline import AnalysisPipeline
import json
import os
import util
import cellplot
import matplotlib.pyplot as plt


def run_script(cell_range):


    # # path_to_data = '/Users/stevecharczynski/workspace/data/brincat_miller'
    # path_to_data = "/usr3/bustaff/scharcz/workspace/brincat_miller/"
    # time_info = RegionInfo(500, 1750, 1.0)
    # data_processor = DataProcessor(path_to_data, cell_range, time_info=time_info)
    # n = 2
    # mean_delta = 0.10 * (time_info.region_high - time_info.region_low)
    # mean_bounds = (
    #     (time_info.region_low - mean_delta),
    #     (time_info.region_high + mean_delta))
    # swarm_params = {
    #                 "phip" : 0.5,
    #                 "phig" : 0.5,
    #                 "omega" : 0.6,
    #                 "minstep" : 1e-10,
    #                 "minfunc" : 1e-10,
    #                 "maxiter" : 1000
    #             }
    # bounds = {
    #             "a_1":[0, 1 / n],
    #             "ut":[mean_bounds[0], mean_bounds[1]], 
    #             "st":[10, 1000], 
    #             "a_0":[10**-10, 1 / n]
    #             }   
    # bounds_c = {"a_0":[10**-10, 0.999]}
    # pipeline = AnalysisPipeline(cell_range, data_processor, ["Const", "Time"], 0, swarm_params)
    # pipeline.set_model_bounds("Time", bounds)
    # pipeline.set_model_bounds("Const", bounds_c)
    # pipeline.fit_all_models(30)
    # pipeline.compare_models("Const", "Time", 0.01)

    # path_to_data = '/Users/stevecharczynski/workspace/maximum_likelihood/python_ready_data'
    path_to_data = "/usr3/bustaff/scharcz/workspace/cromer/"
    time_info = RegionInfo(400, 2000, 1)
    data_processor = DataProcessor(path_to_data, cell_range, time_info=time_info)
    n_c=5
    swarm_params = {
                "phip" : 0.5,
                "phig" : 0.5,
                "omega" : 0.5,
                "minstep" : 1e-10,
                "minfunc" : 1e-10,
                "maxiter" : 1000
            }
    bounds_cat = {
            "ut":[0, 2400],
            "st":[10, 5000],
            "a_0":[10**-10, 1 / n_c], 
            "a_1":[10**-10, 1 / n_c],
            "a_2":[10**-10, 1 / n_c],
            "a_3":[10**-10, 1 / n_c],
            "a_4":[10**-10, 1 / n_c]
    } 
    n_t = 2
    bounds_t = {
        "a_1":[0, 1 / n_t],
        "ut":[0, 2400], 
        "st":[10, 5000], 
        "a_0":[10**-10, 1 / n_t]
    }
    # bounds_cat = ((0,2400), (10, 5000), (10**-10, 1 / n), (0, 1 / n),(0, 1 / n), (0, 1 / n), (0, 1 / n))
    # bounds= ((0, 1 / n), (0,2400), (10, 5000), (10**-10, 1 / n))
    pipeline = AnalysisPipeline(cell_range, data_processor, ["CatTime", "Time"], 0, swarm_params)
    pipeline.set_model_bounds("Time", bounds_t)
    pipeline.set_model_bounds("CatTime", bounds_cat)
    pipeline.fit_all_models(3)
    pipeline.compare_models("Time", "CatTime", 0.01)
    pipeline.show_condition_fit("CatTime")


    # path_to_data = '/Users/stevecharczynski/workspace/rui_fake_cells/mixed_firing'
    # time_info = RegionInfo(0, 2000, 1.0)
    # data_processor = DataProcessor(path_to_data, cell_range, time_info=time_info)
    # n = 2
    # bounds = {
    #         "a_1":[0, 1 / n],
    #         "ut":[-500, 2500], 
    #         "st":[0.01, 5000], 
    #         "a_0":[10**-10, 1 / n]
    #         }   
    # bounds_c = {"a_0":[10**-10, 0.99]}
    # swarm_params = {
    #                 "phip" : 0.5,
    #                 "phig" : 0.5,
    #                 "omega" : 0.5,
    #                 "minstep" : 1e-10,
    #                 "minfunc" : 1e-10,
    #                 "maxiter" : 1000
    #             }
    # pipeline = AnalysisPipeline(cell_range, data_processor, ["Const", "Time"], 0, swarm_params)
    # pipeline.set_model_bounds("Time", bounds)
    # pipeline.set_model_bounds("Const", bounds_c)
    # pipeline.fit_all_models(3)
    # pipeline.compare_models("Const", "Time", 0.01)

    
    util.collect_data(cell_range, "log_likelihoods")
    util.collect_data(cell_range, "model_comparisons")
    util.collect_data(cell_range, "cell_fits")
# run_script(range(45,46))
if __name__ == "__main__":
    cell_range = sys.argv[-2:]
    cell_range = list(map(int, cell_range))
    cell_range = range(cell_range[0], cell_range[1]+1)
    run_script(cell_range)