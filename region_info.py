import numpy as np


class RegionInfo(object):
    def __init__(self, region_low, region_high):
        self.region_low = region_low
        self.region_high = region_high
        self.region_bin = 1.0
        self.total_bins = self.calc_bins()
        self.converted = False

    def calc_bins(self):
        return len(np.arange(
            self.region_low,
            self.region_high,
            self.region_bin))

    