import sys
from python_data import DataProcessor
from region_info import RegionInfo
from analysis_pipeline import AnalysisPipeline
import util


def run_script(cell_range):

    path_to_data = '/Users/stevecharczynski/workspace/data/brincat_miller'
    time_info = RegionInfo(500, 1750)
    data_processor = DataProcessor(path_to_data, cell_range, time_info=time_info)
    n = 2
    mean_delta = 0.10 * (time_info.region_high - time_info.region_low)
    mean_bounds = (
        (time_info.region_low - mean_delta),
        (time_info.region_high + mean_delta))
    swarm_params = {
                    "phip" : 0.5,
                    "phig" : 0.6,
                    "omega" : 0.6,
                    "minstep" : 1e-10,
                    "minfunc" : 1e-10,
                    "maxiter" : 1000
                }
    bounds = ((0, 1 / n), mean_bounds, (10, 1000), (10**-10, 1 / n))
    bounds_c = [(10**-10, 0.99)]
    pipeline = AnalysisPipeline(cell_range, data_processor, ["Const", "Time"], 0, swarm_params)
    pipeline.set_model_bounds("Time", bounds)
    pipeline.set_model_bounds("Const", bounds_c)
    pipeline.fit_all_models(1)
    pipeline.compare_models("Const", "Time", 0.01)

    util.collect_data(cell_range, "log_likelihoods")
    util.collect_data(cell_range, "model_comparisons")
    util.collect_data(cell_range, "cell_fits")

if __name__ == "__main__":
    cell_range = sys.argv[-2:]
    cell_range = list(map(int, cell_range))
    cell_range = range(cell_range[0], cell_range[1]+1)
    run_script(cell_range)
