# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/master/docs/BENCHMARKS.md
import collections

import numpy as np
from qtpy.QtWidgets import QApplication

import napari


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
        Event = collections.namedtuple('Event', 'is_dragging')
        self.event = Event(is_dragging=True)

    def teardown(self):
        self.viewer.window.close()

    def time_zoom(self):
        """Time to zoom in and zoom out."""
        self.viewer.window.qt_viewer.view.camera.zoom(0.5, center=(0.5, 0.5))
        self.viewer.window.qt_viewer.view.camera.zoom(2.0, center=(0.5, 0.5))

    def time_set_view_slice(self):
        """Time to set view slice."""
        self.layer._set_view_slice()

    def time_refresh(self, n):
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
        self.layer._raw_to_displayed(self.layer._data_raw)

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
        self.layer.on_mouse_move(self.event)
