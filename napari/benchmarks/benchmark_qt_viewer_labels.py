# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/main/docs/BENCHMARKS.md
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List

import numpy as np
from qtpy.QtWidgets import QApplication
from skimage.morphology import diamond, octahedron

import napari
from napari.components.viewer_model import ViewerModel
from napari.qt import QtViewer

from .utils import Skiper


@dataclass
class MouseEvent:
    # mock mouse event class
    type: str
    is_dragging: bool
    pos: List[int]
    view_direction: List[int]


class QtViewerSingleLabelsSuite:
    """Benchmarks for editing a single labels layer in the viewer."""

    def setup(self):
        _ = QApplication.instance() or QApplication([])
        np.random.seed(0)
        self.data = np.random.randint(10, size=(512, 512))
        self.viewer = napari.view_labels(self.data)
        self.layer = self.viewer.layers[0]
        self.layer.brush_size = 10
        self.layer.mode = 'paint'
        self.layer.selected_label = 3
        self.layer._last_cursor_coord = (511, 511)
        self.event = MouseEvent(
            type='mouse_move',
            is_dragging=True,
            pos=[500, 500],
            view_direction=None,
        )

    def teardown(self):
        self.viewer.window.close()

    def time_zoom(self):
        """Time to zoom in and zoom out."""
        self.viewer.window._qt_viewer.view.camera.zoom(0.5, center=(0.5, 0.5))
        self.viewer.window._qt_viewer.view.camera.zoom(2.0, center=(0.5, 0.5))

    def time_set_view_slice(self):
        """Time to set view slice."""
        self.layer._set_view_slice()

    def time_refresh(self):
        """Time to refresh view."""
        self.layer.refresh()

    def time_update_thumbnail(self):
        """Time to update thumbnail."""
        self.layer._update_thumbnail()

    def time_get_value(self):
        """Time to get current value."""
        self.layer.get_value((0,) * 2)

    def time_raw_to_displayed(self):
        """Time to convert raw to displayed."""
        self.layer._raw_to_displayed(self.layer._slice.image.raw)

    def time_paint(self):
        """Time to paint."""
        self.layer.paint((0,) * 2, self.layer.selected_label)

    def time_fill(self):
        """Time to fill."""
        self.layer.fill(
            (0,) * 2,
            1,
            self.layer.selected_label,
        )

    def time_on_mouse_move(self):
        """Time to drag paint on mouse move."""
        self.viewer.window._qt_viewer.canvas._on_mouse_move(self.event)


@lru_cache
def setup_rendering_data(radius, dtype):
    if radius < 1000:
        data = octahedron(radius=radius, dtype=dtype)
    else:
        data = np.zeros((radius // 50, radius * 2, radius * 2), dtype=dtype)
        for i in range(1, data.shape[0] // 2):
            part = diamond(radius=i * 100, dtype=dtype)
            shift = (data.shape[1] - part.shape[0]) // 2
            data[i, shift : -shift - 1, shift : -shift - 1] = part
            data[-i - 1, shift : -shift - 1, shift : -shift - 1] = part

    count = np.count_nonzero(data)
    data[data > 0] = np.random.randint(
        1, min(2000, np.iinfo(dtype).max), size=count, dtype=dtype
    )

    return data


class LabelRendering:
    """Benchmarks for rendering the Labels layer."""

    param_names = ["radius", "dtype", "mode"]
    params = (
        [10, 30, 300, 1500],
        [np.uint8, np.uint16, np.uint32],
        ["auto"],  # "direct"],
    )
    if "GITHUB_ACTIONS" in os.environ:
        skip_params = Skiper(lambda x: x[0] > 20)
    if "PR" in os.environ:
        skip_params = Skiper(lambda x: x[0] > 20)

    def setup(self, radius, dtype, label_mode):
        self.steps = 4 if "GITHUB_ACTIONS" in os.environ else 10
        self.app = QApplication.instance() or QApplication([])
        self.data = setup_rendering_data(radius, dtype)
        scale = self.data.shape[-1] / np.array(self.data.shape)
        self.viewer = ViewerModel()
        self.qt_viewr = QtViewer(self.viewer)
        self.layer = self.viewer.add_labels(self.data, scale=scale)
        self.qt_viewr.show()

    @staticmethod
    def teardown(self, *_):
        if hasattr(self, "viewer"):
            self.qt_viewr.close()

    def _time_iterate_components(self, *_):
        """Time to iterate over components."""
        self.layer.show_selected_label = True
        for i in range(0, 201, (200 // self.steps) or 1):
            self.layer.selected_label = i
            self.app.processEvents()

    def _time_zoom_change(self, *_):
        """Time to zoom in and zoom out."""
        initial_zoom = self.viewer.camera.zoom
        self.viewer.camera.zoom = 0.5 * initial_zoom
        self.app.processEvents()
        self.viewer.camera.zoom = 2 * initial_zoom
        self.app.processEvents()


class LabelRenderingSuite2D(LabelRendering):
    def setup(self, radius, dtype, label_mode):
        super().setup(radius, dtype, label_mode)
        self.viewer.dims.ndisplay = 2
        self.app.processEvents()

    def time_iterate_over_z(self, *_):
        """Time to render the layer."""
        z_size = self.data.shape[0]
        for i in range(0, z_size, z_size // (self.steps * 2)):
            self.viewer.dims.set_point(0, i)
            self.app.processEvents()

    def time_load_3d(self, *_):
        """Time to first render of the layer in 3D."""
        self.app.processEvents()
        self.viewer.dims.ndisplay = 3
        self.app.processEvents()
        self.viewer.dims.ndisplay = 2
        self.app.processEvents()

    def time_iterate_components(self, *args):
        self._time_iterate_components(*args)

    def time_zoom_change(self, *args):
        self._time_zoom_change(*args)


class LabelRenderingSuite3D(LabelRendering):
    def setup(self, radius, dtype, label_mode):
        super().setup(radius, dtype, label_mode)
        self.viewer.dims.ndisplay = 3
        self.app.processEvents()

    def time_rotate(self, *_):
        """Time to rotate the layer."""
        for i in range(0, (self.steps * 20), 5):
            self.viewer.camera.angles = (0, i / 2, i)
            self.app.processEvents()

    def time_iterate_components(self, *args):
        self._time_iterate_components(*args)

    def time_zoom_change(self, *args):
        self._time_zoom_change(*args)
