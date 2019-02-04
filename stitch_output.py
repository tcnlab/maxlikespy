import util
import sys


cell_range = sys.argv[-2:]
cell_range = list(map(int, cell_range))
cell_range[1] += 1

util.collect_data(cell_range, "log_likelihoods")
util.collect_data(cell_range, "model_comparisons")
util.collect_data(cell_range, "cell_fits")