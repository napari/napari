# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/master/docs/BENCHMARKS.md
import numpy as np

from napari.layers import Image


class Image2DSuite:
    """Benchmarks for the Image layer with 2D data."""

    params = [2 ** i for i in range(4, 13)]

    def setup(self, n):
        np.random.seed(0)
        self.data = np.random.random((n, n))
        self.new_data = np.random.random((n, n))
        self.layer = Image(self.data)

    def time_create_layer(self, n):
        """Time to create an image layer."""
        Image(self.data)

    def time_set_view_slice(self, n):
        """Time to set view slice."""
        self.layer._set_view_slice()

    def time_update_thumbnail(self, n):
        """Time to update thumbnail."""
        self.layer._update_thumbnail()

    def time_get_value(self, n):
        """Time to get current value."""
        self.layer.get_value((0,) * 2)

    def time_set_data(self, n):
        """Time to get current value."""
        self.layer.data = self.new_data

    def time_refresh(self, n):
        """Time to refresh view."""
        self.layer.refresh()

    def mem_layer(self, n):
        """Memory used by layer."""
        return self.layer

    def mem_data(self, n):
        """Memory used by raw data."""
        return self.data


class Image3DSuite:
    """Benchmarks for the Image layer with 3D data."""

    params = [2 ** i for i in range(4, 11)]

    def setup(self, n):
        np.random.seed(0)
        self.data = np.random.random((n, n, n))
        self.new_data = np.random.random((n, n, n))
        self.layer = Image(self.data)

    def time_create_layer(self, n):
        """Time to create an image layer."""
        Image(self.data)

    def time_set_view_slice(self, n):
        """Time to set view slice."""
        self.layer._set_view_slice()

    def time_update_thumbnail(self, n):
        """Time to update thumbnail."""
        self.layer._update_thumbnail()

    def time_get_value(self, n):
        """Time to get current value."""
        self.layer.get_value((0,) * 3)

    def time_set_data(self, n):
        """Time to get current value."""
        self.layer.data = self.new_data

    def time_refresh(self, n):
        """Time to refresh view."""
        self.layer.refresh()

    def mem_layer(self, n):
        """Memory used by layer."""
        return Image(self.data)

    def mem_data(self, n):
        """Memory used by raw data."""
        return self.data
