# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/master/docs/BENCHMARKS.md
import numpy as np

from napari.layers import Vectors


class Vectors2DSuite:
    """Benchmarks for the Vectors layer with 2D data"""

    params = [2 ** i for i in range(4, 18, 2)]

    def setup(self, n):
        np.random.seed(0)
        self.data = np.random.random((n, 2, 2))
        self.layer = Vectors(self.data)

    def time_create_layer(self, n):
        """Time to create an image layer."""
        Vectors(self.data)

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
        self.layer.get_value((0,) * 2)

    def time_width(self, n):
        """Time to update width."""
        self.layer.width = 2

    def time_length(self, n):
        """Time to update length."""
        self.layer.length = 2

    def mem_layer(self, n):
        """Memory used by layer."""
        return self.layer

    def mem_data(self, n):
        """Memory used by raw data."""
        return self.data


class Vectors3DSuite:
    """Benchmarks for the Vectors layer with 3D data."""

    params = [2 ** i for i in range(4, 18, 2)]

    def setup(self, n):
        np.random.seed(0)
        self.data = np.random.random((n, 2, 3))
        self.layer = Vectors(self.data)

    def time_create_layer(self, n):
        """Time to create a layer."""
        Vectors(self.data)

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
        self.layer.get_value((0,) * 3)

    def time_width(self, n):
        """Time to update width."""
        self.layer.width = 2

    def time_length(self, n):
        """Time to update length."""
        self.layer.length = 2

    def mem_layer(self, n):
        """Memory used by layer."""
        return self.layer

    def mem_data(self, n):
        """Memory used by raw data."""
        return self.data
