import numpy as np
from ..util.event import EmitterGroup


class Histogram:
    """Histogram Model, numpy API with vispy events"""

    def __init__(
        self, data=None, bins=None, range=None, weights=None, density=None
    ):
        self.bins = bins or 'fd'  # (Freedman Diaconis Estimator)
        self.range = range  # numpy also shadows range, we mimick their API
        self.weights = weights
        self.density = density
        self.counts = None
        self.bin_edges = None

        # Events:
        self.events = EmitterGroup(source=self, data=None)

        if data is not None:
            self.set_data(data)

    def set_data(self, data):
        print("calculating histogram")
        self.counts, self.bin_edges = np.histogram(
            np.asarray(data).ravel(),
            bins=self.bins,
            range=self.range,
            weights=self.weights,
            density=self.density,
        )
        self.events.data(counts=self.counts, bin_edges=self.bin_edges)
