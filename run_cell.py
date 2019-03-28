import sys
from python_data import DataProcessor
from analysis_pipeline import AnalysisPipeline
import json
import os
import util
import cellplot
import matplotlib.pyplot as plt


def run_script(cell_range):

    # path_to_data = '/Users/stevecharczynski/workspace/data/brincat_miller'
    # # path_to_data = "/projectnb/ecog-eeg/stevechar/data/brincat_miller/"
    # time_info = [500, 1750]
    # data_processor = DataProcessor(
    #     path_to_data, cell_range, time_info=time_info)
    # n = 2
    # mean_delta = 0.10 * (time_info[1] - time_info[0])
    # mean_bounds = (
    #     (time_info[0] - mean_delta),
    #     (time_info[1] + mean_delta))
    # swarm_params = {
    #     "phip": 0.5,
    #     "phig": 0.5,
    #     "omega": 0.6,
    #     "minstep": 1e-10,
    #     "minfunc": 1e-10,
    #     "maxiter": 1000 
    # }
    # bounds = {
    #     "a_1": [0, 1 / n],
    #     "ut": [mean_bounds[0], mean_bounds[1]],
    #     "st": [10, 1000],
    #     "a_0": [10**-10, 1 / n]
    # }
    # bounds_c = {"a_0": [10**-10, 0.999]}
    # pipeline = AnalysisPipeline(cell_range, data_processor, [
    #                             "Const", "Time"], 0)
    # pipeline.set_model_bounds("Time", bounds)
    # pipeline.set_model_bounds("Const", bounds_c)
    # pipeline.fit_all_models(swarm_params)
    # pipeline.compare_models("Const", "Time", 0.01)

    # path_to_data = "/Users/stevecharczynski/workspace/data/salz"
    # # path_to_data = "/projectnb/ecog-eeg/stevechar/data/salz"
    # time_info = [1000, 21000]
    # data_processor = DataProcessor(
    #     path_to_data, cell_range, time_info=time_info)
    # n = 2
    # mean_delta = 0.10 * (time_info[1] - time_info[0])
    # mean_bounds = (
    #     (time_info[0] - mean_delta),
    #     (time_info[1] + mean_delta))
    # solver_params = {
    #     "niter": 200,
    #     "stepsize": 100,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    # }
    # bounds = {
    #     "a_1": [0, 1 / n],
    #     "ut": [mean_bounds[0], mean_bounds[1]],
    #     "st": [10, 50000],
    #     "a_0": [10**-10, 1 / n]
    # }
    # bounds_c = {"a_0": [10**-10, 0.999]}
    # pipeline = AnalysisPipeline(cell_range, data_processor, [
    #                             "Const", "Time"], 0)
    # pipeline.set_model_bounds("Time", bounds)
    # pipeline.set_model_bounds("Const", bounds_c)
    # pipeline.fit_all_models(solver_params)
    # pipeline.compare_models("Const", "Time", 0.01)

    # path_to_data = "/Users/stevecharczynski/workspace/data/cromer"
    path_to_data = '/projectnb/ecog-eeg/stevechar/data/cromer'
    # path_to_data = "/usr3/bustaff/scharcz/workspace/cromer/"
    time_info = [400, 2000]
    data_processor = DataProcessor(
        path_to_data, cell_range, time_info=time_info)
    n_c = 5
    solver_params = {
        "niter": 50,
        "stepsize": 100,
        "interval": 10,
        "method": "TNC",
        "use_jac": True,
    }
    bounds_cat = {
        "ut": [0, 2400],
        "st": [10, 5000],
        "a_0": [10**-10, 1 / n_c],
        "a_1": [10**-10, 1 / n_c],
        "a_2": [10**-10, 1 / n_c],
        "a_3": [10**-10, 1 / n_c],
        "a_4": [10**-10, 1 / n_c]
    }
    n_cs = 3
    bounds_cs = {
        "ut": [0, 2400],
        "st": [10, 5000],
        "a_0": [10**-10, 1 / n_cs],
        "a_1": [10**-10, 1 / n_cs],
        "a_2": [10**-10, 1 / n_cs],
    }
    pipeline = AnalysisPipeline(cell_range, data_processor, [
                                "CatTime", "CatSetTime"], 0)
    pipeline.set_model_bounds("CatSetTime", bounds_cs)
    pipeline.set_model_bounds("CatTime", bounds_cat)
    pipeline.set_model_info("CatSetTime", "pairs", [(1,2), (3,4)])
    pipeline.fit_all_models(solver_params=solver_params)
    pipeline.compare_models("CatSetTime", "CatTime", 0.01)
    pipeline.show_condition_fit("CatTime")
    pipeline.show_condition_fit("CatSetTime")

    # path_to_data = "/Users/stevecharczynski/workspace/data/kim"
    # path_to_data = "/projectnb/ecog-eeg/stevechar/data/kim"
    # time_info = [0, 4784]
    # data_processor = DataProcessor(
    #     path_to_data, cell_range, time_info=time_info)
    # n = 2
    # swarm_params = {
    #     "phip": 0.5,
    #     "phig": 0.5,
    #     "omega": 0.5,
    #     "minstep": 1e-10,
    #     "minfunc": 1e-10,
    #     "maxiter": 1000
    # }
    # bounds = {
    #     "a_1": [0, 1 / n],
    #     "ut": [-200, 5200],
    #     "st": [10, 10000],
    #     "a_0": [10**-10, 1 / n]
    # }
    # bounds_c = {"a_0": [10**-10, 0.999]}
    # pipeline = AnalysisPipeline(cell_range, data_processor, [
    #                             "Const", "Time"], 0, swarm_params)
    # pipeline.set_model_bounds("Time", bounds)
    # pipeline.set_model_bounds("Const", bounds_c)
    # pipeline.fit_all_models(1)
    # pipeline.compare_models("Const", "Time", 0.01)

    # path_to_data = "/Users/stevecharczynski/workspace/data/cromer"
    # # path_to_data =  "/projectnb/ecog-eeg/stevechar/data/cromer"
    # time_info = [400, 2000]
    # data_processor = DataProcessor(
    #     path_to_data, cell_range, time_info=time_info)
    # n_t = 2.
    # swarm_params = {
    #     "swarmsize": 50,
    #     "phip": 0.5,
    #     "phig": 0.5,
    #     "omega": 0.5,
    #     "minstep": 1e-10,
    #     "minfunc": 1e-10,
    #     "maxiter": 1000
    # }
    # bounds = {
    #     "a_1": [10**-10, 1 / n_t],
    #     "ut": [0., 2400.],
    #     "st": [10., 5000.],
    #     "a_0": [10**-10, 1 / n_t]
    # }
    # bounds_c = {"a_0": [10**-10, 0.999]}
    # pipeline = AnalysisPipeline(cell_range, data_processor, [
    #                             "Time", "Const"], 0, swarm_params)
    # pipeline.set_model_bounds("Time", bounds)
    # pipeline.set_model_bounds("Const", bounds_c)
    # pipeline.fit_all_models(1)
    # pipeline.compare_models("Const", "Time", 0.01)


    # path_to_data = '/Users/stevecharczynski/workspace/data/cromer'
    # # path_to_data = "/usr3/bustaff/scharcz/workspace/cromer/"
    # time_info = [400, 2000]
    # data_processor = DataProcessor(
    #     path_to_data, cell_range, time_info=time_info)
    # n_c = 3
    # swarm_params = {
    #     "phip": 0.5,
    #     "phig": 0.7,
    #     "omega": 0.7,
    #     "minstep": 1e-10,
    #     "minfunc": 1e-10,
    #     "maxiter": 1000
    # }
    # bounds_cat = {
    #     "ut": [0, 2400],
    #     "st": [10, 5000],
    #     "a_0": [10**-10, 1 / n_c],
    #     "a_1": [10**-10, 1 / n_c],
    #     "a_2": [10**-10, 1 / n_c],
    # }
    # n_t = 2
    # bounds_t = {
    #     "a_1": [0, 1 / n_t],
    #     "ut": [0, 2400],
    #     "st": [10, 5000],
    #     "a_0": [10**-10, 1 / n_t]
    # }
    # # bounds_cat = ((0,2400), (10, 5000), (10**-10, 1 / n), (0, 1 / n),(0, 1 / n), (0, 1 / n), (0, 1 / n))
    # # bounds= ((0, 1 / n), (0,2400), (10, 5000), (10**-10, 1 / n))
    # pipeline = AnalysisPipeline(cell_range, data_processor, [
    #                             "CatSetTime", "Time"], 0, swarm_params)
    # pipeline.set_model_bounds("Time", bounds_t)
    # pipeline.set_model_bounds("CatSetTime", bounds_cat)
    # pipeline.set_model_info("CatSetTime", "pairs", [(1,2), (3,4)])
    # pipeline.fit_all_models(1)
    # pipeline.compare_models("Time", "CatSetTime", 0.01)
    # pipeline.show_condition_fit("CatSetTime")

    # path_to_data = '/Users/stevecharczynski/workspace/rui_fake_cells/mixed_firing'
    # time_info = RegionInfo(0, 2000)
    # data_processor = DataProcessor(
    #     path_to_data, cell_range, time_info=time_info)
    # n = 2
    # bounds = {
    #     "a_1": [0, 1 / n],
    #     "ut": [-500, 2500],
    #     "st": [0.01, 5000],
    #     "a_0": [10**-10, 1 / n]
    # }
    # bounds_c = {"a_0": [10**-10, 0.99]}
    # swarm_params = {
    #     "phip": 0.5,
    #     "phig": 0.5,
    #     "omega": 0.5,
    #     "minstep": 1e-10,
    #     "minfunc": 1e-10,
    #     "maxiter": 1000
    # }
    # pipeline = AnalysisPipeline(cell_range, data_processor, [
    #                             "Const", "Time"], 0, swarm_params)
    # pipeline.set_model_bounds("Time", bounds)
    # pipeline.set_model_bounds("Const", bounds_c)
    # pipeline.fit_all_models(3)
    # pipeline.compare_models("Const", "Time", 0.01)

    # util.collect_data(cell_range, "log_likelihoods")
    # util.collect_data(cell_range, "model_comparisons")
    # util.collect_data(cell_range, "cell_fits")


# run_script(range(0,2))
if __name__ == "__main__":
    cell_range = sys.argv[-2:]
    cell_range = list(map(int, cell_range))
    cell_range = range(cell_range[0], cell_range[1]+1)
    run_script(cell_range)
