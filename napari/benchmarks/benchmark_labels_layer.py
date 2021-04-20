# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/master/docs/BENCHMARKS.md
import numpy as np

from napari.layers import Labels


class Labels2DSuite:
    """Benchmarks for the Labels layer with 2D data"""

    params = [2 ** i for i in range(4, 13)]

    def setup(self, n):
        np.random.seed(0)
        self.data = np.random.randint(20, size=(n, n))
        self.layer = Labels(self.data)

    def time_create_layer(self, n):
        """Time to create layer."""
        Labels(self.data)

    def time_set_view_slice(self, n):
        """Time to set view slice."""
        self.layer._set_view_slice()

    def time_refresh(self, n):
        """Time to refresh view."""
        self.layer.refresh()

    def time_update_thumbnail(self, n):
        """Time to update thumbnail."""
        self.layer._update_thumbnail()

    def time_get_value(self, n):
        """Time to get current value."""
        self.layer.get_value((0,) * 2)

    def time_raw_to_displayed(self, n):
        """Time to convert raw to displayed."""
        self.layer._raw_to_displayed(self.layer._data_raw)

    def time_paint_square(self, n):
        """Time to paint square."""
        self.layer.brush_shape = 'square'
        self.layer.paint((0,) * 2, self.layer.selected_label)

    def time_paint_circle(self, n):
        """Time to paint circle."""
        self.layer.brush_shape = 'circle'
        self.layer.paint((0,) * 2, self.layer.selected_label)

    def time_fill(self, n):
        """Time to fill."""
        self.layer.fill(
            (0,) * 2,
            1,
            self.layer.selected_label,
        )

    def mem_layer(self, n):
        """Memory used by layer."""
        return self.layer

    def mem_data(self, n):
        """Memory used by raw data."""
        return self.data


class Labels3DSuite:
    """Benchmarks for the Labels layer with 3D data."""

    params = [2 ** i for i in range(4, 11)]

    def setup(self, n):
        np.random.seed(0)
        self.data = np.random.randint(20, size=(n, n, n))
        self.layer = Labels(self.data)

    def time_create_layer(self, n):
        """Time to create layer."""
        Labels(self.data)

    def time_set_view_slice(self, n):
        """Time to set view slice."""
        self.layer._set_view_slice()

    def time_refresh(self, n):
        """Time to refresh view."""
        self.layer.refresh()

    def time_update_thumbnail(self, n):
        """Time to update thumbnail."""
        self.layer._update_thumbnail()

    def time_get_value(self, n):
        """Time to get current value."""
        self.layer.get_value((0,) * 3)

    def time_raw_to_displayed(self, n):
        """Time to convert raw to displayed."""
        self.layer._raw_to_displayed(self.layer._data_raw)

    def time_paint_square(self, n):
        """Time to paint square."""
        self.layer.brush_shape = 'square'
        self.layer.paint((0,) * 3, self.layer.selected_label)

    def time_paint_circle(self, n):
        """Time to paint circle."""
        self.layer.brush_shape = 'circle'
        self.layer.paint((0,) * 3, self.layer.selected_label)

    def time_fill(self, n):
        """Time to fill."""
        self.layer.fill(
            (0,) * 3,
            1,
            self.layer.selected_label,
        )

    def mem_layer(self, n):
        """Memory used by layer."""
        return self.layer

    def mem_data(self, n):
        """Memory used by raw data."""
        return self.data
