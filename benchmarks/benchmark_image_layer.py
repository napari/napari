# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/master/BENCHMARKS.md
import numpy as np
from napari.layers import Image


class ImageSuite:
    """Benchmarks for the Image layer."""

    def setup(self):
        shape = (10, 10)
        np.random.seed(0)
        self.data = np.random.random(shape)

    def time_create_layer(self):
        """Time to create an image layer."""
        layer = Image(self.data)
