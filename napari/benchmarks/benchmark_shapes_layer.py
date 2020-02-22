# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/master/docs/BENCHMARKS.md
import numpy as np
from napari.layers import Shapes


class Shapes2DSuite:
    """Benchmarks for the Shapes layer with 2D data"""

    params = [2 ** i for i in range(4, 9)]

    def setup(self, n):
        np.random.seed(0)
        self.data = [50 * np.random.random((6, 2)) for i in range(n)]
        self.layer = Shapes(self.data, shape_type='polygon')

    def time_create_layer(self, n):
        """Time to create an image layer."""
        Shapes(self.data, shape_type='polygon')

    def time_refresh(self, n):
        """Time to refresh view."""
        self.layer.refresh()

    def time_set_view_slice(self, n):
        """Time to set view slice."""
        self.layer._set_view_slice()

    def time_update_thumbnail(self, n):
        """Time to update thumbnail."""
        self.layer._update_thumbnail()

    def time_get_value(self, n):
        """Time to get current value."""
        self.layer.get_value()

    def mem_layer(self, n):
        """Memory used by layer."""
        return self.layer

    def mem_data(self, n):
        """Memory used by raw data."""
        return self.data


class Shapes3DSuite:
    """Benchmarks for the Shapes layer with 3D data."""

    params = [2 ** i for i in range(4, 9)]

    def setup(self, n):
        np.random.seed(0)
        self.data = [50 * np.random.random((6, 3)) for i in range(n)]
        self.layer = Shapes(self.data, shape_type='polygon')

    def time_create_layer(self, n):
        """Time to create a layer."""
        Shapes(self.data, shape_type='polygon')

    def time_refresh(self, n):
        """Time to refresh view."""
        self.layer.refresh()

    def time_set_view_slice(self, n):
        """Time to set view slice."""
        self.layer._set_view_slice()

    def time_update_thumbnail(self, n):
        """Time to update thumbnail."""
        self.layer._update_thumbnail()

    def time_get_value(self, n):
        """Time to get current value."""
        self.layer.get_value()

    def mem_layer(self, n):
        """Memory used by layer."""
        return self.layer

    def mem_data(self, n):
        """Memory used by raw data."""
        return self.data
