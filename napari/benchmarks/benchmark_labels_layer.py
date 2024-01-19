# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/main/docs/BENCHMARKS.md
import inspect
import os
from copy import copy
from unittest.mock import patch

import numpy as np

from napari.components.dims import Dims
from napari.layers import Labels
from napari.utils.colormaps import direct_colormap
from napari.utils.colormaps.colormap import (
    _cast_labels_data_to_texture_dtype_direct,
)

from .utils import Skiper, labeled_particles

MAX_VAL = 2**23


def _warm_numba(dkt, dtype):
    cmap = direct_colormap(dkt)
    data1 = np.empty((10, 10), dtype=dtype)
    data2 = np.empty((10,), dtype=dtype)
    for _ in range(3):
        with patch(
            'napari.utils.colormaps.colormap.SINGLE_THREAD_THRESHOLD',
            500,
            create=True,
        ):
            _cast_labels_data_to_texture_dtype_direct(data1, cmap)
            _cast_labels_data_to_texture_dtype_direct(data2, cmap)
        with patch(
            'napari.utils.colormaps.colormap.SINGLE_THREAD_THRESHOLD',
            5,
            create=True,
        ):
            _cast_labels_data_to_texture_dtype_direct(data1, cmap)
            _cast_labels_data_to_texture_dtype_direct(data2, cmap)


class Labels2DSuite:
    """Benchmarks for the Labels layer with 2D data"""

    param_names = ['n', 'dtype']
    params = ([2**i for i in range(4, 13)], [np.uint8, np.int32])

    if "PR" in os.environ:
        skip_params = Skiper(lambda x: x[0] > 2**5)

    def setup(self, n, dtype):
        np.random.seed(0)
        self.data = labeled_particles(
            (n, n), dtype=dtype, n=int(np.log2(n) ** 2), seed=1
        )
        self.layer = Labels(self.data)
        self.layer._raw_to_displayed(self.data, (slice(0, n), slice(0, n)))

    def time_create_layer(self, *_):
        """Time to create layer."""
        Labels(self.data)

    def time_set_view_slice(self, *_):
        """Time to set view slice."""
        self.layer._set_view_slice()

    def time_refresh(self, *_):
        """Time to refresh view."""
        self.layer.refresh()

    def time_update_thumbnail(self, *_):
        """Time to update thumbnail."""
        self.layer._update_thumbnail()

    def time_get_value(self, *_):
        """Time to get current value."""
        self.layer.get_value((0,) * 2)

    def time_raw_to_displayed_full(self, *_):
        """Time to convert raw to displayed."""
        self.layer._slice.image.raw[0, :] += 1  # simulate changes
        self.layer._raw_to_displayed(self.layer._slice.image.raw)

    def time_raw_to_displayed_sliced(self, *_):
        self.layer._slice.image.raw[0, :] += 1  # simulate changes
        self.layer._raw_to_displayed(
            self.layer._slice.image.raw, (slice(0, 10), slice(0, 10))
        )

    def time_raw_to_displayed_sliced_num(self, *_):
        self.layer._slice.image.raw[0, :] = 1  # simulate changes
        if inspect.signature(self.layer._raw_to_displayed).parameters.get(
            'num_colors'
        ):
            self.layer._raw_to_displayed(
                self.layer._slice.image.raw, (slice(0, 10), slice(0, 10)), 1
            )
        else:
            self.layer._raw_to_displayed(
                self.layer._slice.image.raw, (slice(0, 10), slice(0, 10))
            )

    def time_paint_circle(self, *_):
        """Time to paint circle."""
        self.layer.mode = "paint"
        self.layer._draw(self.layer.selected_label, (0, 0), (0, 0))

    def time_paint_circle_mult(self, n, *_):
        """Time to paint circle."""
        self.layer.mode = "paint"
        self.layer._draw(self.layer.selected_label, (0, 0), (0, 0))
        self.layer._draw(
            self.layer.selected_label, (n // 2, n // 2), (n // 2, n // 2)
        )
        self.layer._draw(
            self.layer.selected_label, (n - 1, n - 1), (n - 1, n - 1)
        )

    def time_paint_line(self, n, *_):
        """Time to paint circle."""
        self.layer.mode = "paint"
        self.layer._draw(self.layer.selected_label, (0, 0), (n - 1, n - 1))

    def time_fill(self, *_):
        """Time to fill."""
        self.layer.fill(
            (0,) * 2,
            1,
            self.layer.selected_label,
        )

    def mem_layer(self, *_):
        """Memory used by layer."""
        return copy(self.layer)

    def mem_data(self, *_):
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
        self.data = labeled_particles(
            (n, n), dtype=np.int32, n=int(np.log2(n) ** 2), seed=1
        )

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
    def setup(self, n, dtype):
        if "PR" in os.environ and n > 32:
            raise NotImplementedError("Skip on PR (speedup)")
        np.random.seed(0)
        info = np.iinfo(dtype)
        self.data = labeled_particles(
            (n, n), dtype=dtype, n=int(np.log2(n) ** 2), seed=1
        )
        random_label_ids = np.random.randint(
            low=max(-10000, info.min), high=min(10000, info.max), size=20
        )
        self.layer = Labels(
            self.data,
            color={i + 1: np.random.random(4) for i in random_label_ids},
        )
        _warm_numba(self.layer.colormap.color_dict, dtype)
        self.layer._raw_to_displayed(
            self.layer._slice.image.raw, (slice(0, n), slice(0, n))
        )


class Labels3DSuite:
    """Benchmarks for the Labels layer with 3D data."""

    param_names = ['n', 'dtype']
    params = ([2**i for i in range(4, 11)], [np.uint8, np.uint32])
    if "PR" in os.environ:
        skip_params = [(2**i,) for i in range(6, 11)]

    def setup(self, n, dtype):
        if "CI" in os.environ and n > 512:
            raise NotImplementedError("Skip on CI (not enough memory)")

        np.random.seed(0)
        self.data = labeled_particles(
            (n, n, n), dtype=dtype, n=int(np.log2(n) ** 2), seed=1
        )
        self.layer = Labels(self.data)
        self.layer._slice_dims(Dims(ndim=3, ndisplay=3))
        self.layer._raw_to_displayed(
            self.layer._slice.image.raw,
            (slice(0, n), slice(0, n), slice(0, n)),
        )

    # @mark.skip_params_if([(2**i,) for i in range(6, 11)], condition="PR" in os.environ)
    def time_create_layer(self, *_):
        """Time to create layer."""
        Labels(self.data)

    def time_set_view_slice(self, *_):
        """Time to set view slice."""
        self.layer._set_view_slice()

    def time_refresh(self, *_):
        """Time to refresh view."""
        self.layer.refresh()

    def time_update_thumbnail(self, *_):
        """Time to update thumbnail."""
        self.layer._update_thumbnail()

    def time_get_value(self, *_):
        """Time to get current value."""
        self.layer.get_value((0,) * 3)

    def time_raw_to_displayed(self, *_):
        """Time to convert raw to displayed."""
        self.layer._slice.image.raw[0, 0, :] += 1  # simulate changes
        self.layer._raw_to_displayed(self.layer._slice.image.raw)

    def time_paint_circle(self, *_):
        """Time to paint circle."""
        self.layer.paint((0,) * 3, self.layer.selected_label)

    def time_fill(self, *_):
        """Time to fill."""
        self.layer.fill(
            (0,) * 3,
            1,
            self.layer.selected_label,
        )

    def mem_layer(self, *_):
        """Memory used by layer."""
        return copy(self.layer)

    def mem_data(self, *_):
        """Memory used by raw data."""
        return self.data
