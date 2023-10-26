# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/main/docs/BENCHMARKS.md
import os

import numpy as np

from napari.components.dims import Dims
from napari.layers import Labels

from .utils import Skiper


class Labels2DSuite:
    """Benchmarks for the Labels layer with 2D data"""

    params = [2**i for i in range(4, 13)]

    if "PR" in os.environ:
        skip_params = [(2**i,) for i in range(6, 13)]

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
        self.layer._slice.image.raw[0, :] += 1  # simulate changes
        self.layer._raw_to_displayed(self.layer._slice.image.raw)

    def time_paint_circle(self, n):
        """Time to paint circle."""
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


class LabelsDrawing2DSuite:
    """Benchmark for brush drawing in the Labels layer with 2D data."""

    param_names = ['n', 'brush_size', 'color_mode', 'contour']
    params = ([512, 3072], [8, 64, 256], ['auto', 'direct'], [0, 1])

    if "PR" in os.environ:
        skip_params = Skiper(lambda x: x[0] > 512 or x[1] > 64)

    def setup(self, n, brush_size, color_mode, contour):
        np.random.seed(0)
        self.data = np.random.randint(64, size=(n, n), dtype=np.int32)

        colors = None
        if color_mode == 'direct':
            random_label_ids = np.random.randint(64, size=50)
            colors = {i + 1: np.random.random(4) for i in random_label_ids}

        self.layer = Labels(self.data, color=colors)

        self.layer.brush_size = brush_size
        self.layer.contour = contour
        self.layer.mode = 'paint'

    def time_draw(self, n, brush_size, color_mode, contour):
        new_label = self.layer._slice.image.raw[0, 0] + 1

        with self.layer.block_history():
            last_coord = (0, 0)
            for x in np.linspace(0, n - 1, num=30)[1:]:
                self.layer._draw(
                    new_label=new_label,
                    last_cursor_coord=last_coord,
                    coordinates=(x, x),
                )
                last_coord = (x, x)


class Labels2DColorDirectSuite(Labels2DSuite):
    def setup(self, n):
        if "PR" in os.environ and n > 32:
            raise NotImplementedError("Skip on PR (speedup)")
        np.random.seed(0)
        self.data = np.random.randint(low=-10000, high=10000, size=(n, n))
        random_label_ids = np.random.randint(low=-10000, high=10000, size=20)
        self.layer = Labels(
            self.data,
            color={i + 1: np.random.random(4) for i in random_label_ids},
        )
        self.layer._raw_to_displayed(self.layer._slice.image.raw)


class Labels3DSuite:
    """Benchmarks for the Labels layer with 3D data."""

    params = [2**i for i in range(4, 11)]
    if "PR" in os.environ:
        skip_params = [(2**i,) for i in range(6, 11)]

    def setup(self, n):
        if "CI" in os.environ and n > 512:
            raise NotImplementedError("Skip on CI (not enough memory)")

        np.random.seed(0)
        self.data = np.random.randint(20, size=(n, n, n))
        self.layer = Labels(self.data)
        self.layer._slice_dims(Dims(ndim=3, ndisplay=3))

    # @mark.skip_params_if([(2**i,) for i in range(6, 11)], condition="PR" in os.environ)
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
        self.layer._slice.image.raw[0, 0, :] += 1  # simulate changes
        self.layer._raw_to_displayed(self.layer._slice.image.raw)

    def time_paint_circle(self, n):
        """Time to paint circle."""
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
