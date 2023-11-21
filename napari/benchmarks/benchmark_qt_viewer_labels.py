# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/main/docs/BENCHMARKS.md
import os
from dataclasses import dataclass
from typing import List

import numpy as np
from qtpy.QtWidgets import QApplication
from skimage.morphology import octahedron

import napari
from napari.benchmarks.utils import Skiper


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
            pos=(500, 500),
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


class LabelRenderingSuite:
    """Benchmarks for rendering the Labels layer."""

    param_names = ["radius", "dtype"]
    params = ([20, 100, 300], [np.uint8, np.uint16, np.uint32])
    if "PR" in os.environ:
        skip_params = Skiper(lambda x: x[0] >= 100)

    def setup(self, radius, dtype):
        self.app = QApplication.instance() or QApplication([])

        self.data = octahedron(radius=radius, dtype=dtype)
        self.viewer = napari.view_labels(self.data)
        self.layer = self.viewer.layers[0]


class LabelRenderingSuite2D(LabelRenderingSuite):
    def setup(self, radius, dtype):
        super().setup(radius, dtype)
        self.viewer.dims.ndisplay = 2

    def time_iterate_over_z(self, radius, dtype):
        """Time to render the layer."""
        for i in range(0, radius * 2, radius // 10):
            self.viewer.dims.set_point(0, i)
            self.app.processEvents()


class LabelRenderingSuite3D(LabelRenderingSuite):
    def setup(self, radius, dtype):
        super().setup(radius, dtype)
        self.viewer.dims.ndisplay = 3

    def time_rotate(self, radius, dtype):
        """Time to rotate the layer."""
        for i in range(0, 360, 5):
            self.viewer.camera.angles = (0, i / 2, i)
            self.app.processEvents()
