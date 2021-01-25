# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/master/docs/BENCHMARKS.md
import numpy as np
from qtpy.QtWidgets import QApplication

import napari


class QtViewerViewImageSuite:
    """Benchmarks for viewing images in the viewer."""

    params = [2 ** i for i in range(4, 13)]

    def setup(self, n):
        _ = QApplication.instance() or QApplication([])
        np.random.seed(0)
        self.data = np.random.random((n, n))
        self.viewer = None

    def teardown(self, n):
        self.viewer.window.close()

    def time_view_image(self, n):
        """Time to view an image."""
        self.viewer = napari.view_image(self.data)


class QtViewerAddImageSuite:
    """Benchmarks for adding images to the viewer."""

    params = [2 ** i for i in range(4, 13)]

    def setup(self, n):
        _ = QApplication.instance() or QApplication([])
        np.random.seed(0)
        self.data = np.random.random((n, n))
        self.viewer = napari.Viewer()

    def teardown(self, n):
        self.viewer.window.close()

    def time_add_image(self, n):
        """Time to view an image."""
        self.viewer.add_image(self.data)


class QtViewerImageSuite:
    """Benchmarks for images in the viewer."""

    params = [2 ** i for i in range(4, 13)]

    def setup(self, n):
        _ = QApplication.instance() or QApplication([])
        np.random.seed(0)
        self.data = np.random.random((n, n))
        self.viewer = napari.view_image(self.data)

    def teardown(self, n):
        self.viewer.window.close()

    def time_zoom(self, n):
        """Time to zoom in and zoom out."""
        self.viewer.window.qt_viewer.view.camera.zoom(0.5, center=(0.5, 0.5))
        self.viewer.window.qt_viewer.view.camera.zoom(2.0, center=(0.5, 0.5))

    def time_refresh(self, n):
        """Time to refresh view."""
        self.viewer.layers[0].refresh()

    def time_set_view_slice(self, n):
        """Time to set view slice."""
        self.viewer.layers[0]._set_view_slice()

    def time_update_thumbnail(self, n):
        """Time to update thumbnail."""
        self.viewer.layers[0]._update_thumbnail()

    def time_get_value(self, n):
        """Time to get current value."""
        self.viewer.layers[0].get_value((0,) * 2)


class QtViewerSingleImageSuite:
    """Benchmarks for a single image layer in the viewer."""

    def setup(self):
        _ = QApplication.instance() or QApplication([])
        np.random.seed(0)
        self.data = np.random.random((128, 128, 128))
        self.new_data = np.random.random((128, 128, 128))
        self.viewer = napari.view_image(self.data)

    def teardown(self):
        self.viewer.window.close()

    def time_zoom(self):
        """Time to zoom in and zoom out."""
        self.viewer.window.qt_viewer.view.camera.zoom(0.5, center=(0.5, 0.5))
        self.viewer.window.qt_viewer.view.camera.zoom(2.0, center=(0.5, 0.5))

    def time_set_data(self):
        """Time to set view slice."""
        self.viewer.layers[0].data = self.new_data

    def time_refresh(self):
        """Time to refresh view."""
        self.viewer.layers[0].refresh()

    def time_set_view_slice(self):
        """Time to set view slice."""
        self.viewer.layers[0]._set_view_slice()

    def time_update_thumbnail(self):
        """Time to update thumbnail."""
        self.viewer.layers[0]._update_thumbnail()

    def time_get_value(self):
        """Time to get current value."""
        self.viewer.layers[0].get_value((0,) * 3)

    def time_ndisplay(self):
        """Time to enter 3D rendering."""
        self.viewer.dims.ndisplay = 3


class QtViewerSingleInvisbleImageSuite:
    """Benchmarks for a invisible single image layer in the viewer."""

    def setup(self):
        _ = QApplication.instance() or QApplication([])
        np.random.seed(0)
        self.data = np.random.random((128, 128, 128))
        self.new_data = np.random.random((128, 128, 128))
        self.viewer = napari.view_image(self.data, visible=False)

    def teardown(self):
        self.viewer.window.close()

    def time_zoom(self):
        """Time to zoom in and zoom out."""
        self.viewer.window.qt_viewer.view.camera.zoom(0.5, center=(0.5, 0.5))
        self.viewer.window.qt_viewer.view.camera.zoom(2.0, center=(0.5, 0.5))

    def time_set_data(self):
        """Time to set view slice."""
        self.viewer.layers[0].data = self.new_data

    def time_refresh(self):
        """Time to refresh view."""
        self.viewer.layers[0].refresh()

    def time_set_view_slice(self):
        """Time to set view slice."""
        self.viewer.layers[0]._set_view_slice()

    def time_update_thumbnail(self):
        """Time to update thumbnail."""
        self.viewer.layers[0]._update_thumbnail()

    def time_get_value(self):
        """Time to get current value."""
        self.viewer.layers[0].get_value((0,) * 3)

    def time_ndisplay(self):
        """Time to enter 3D rendering."""
        self.viewer.dims.ndisplay = 3


class QtImageRenderingSuite:
    """Benchmarks for a single image layer in the viewer."""

    params = [2 ** i for i in range(4, 13)]

    def setup(self, n):
        _ = QApplication.instance() or QApplication([])
        np.random.seed(0)
        self.data = np.random.random((n, n)) * 2 ** 12
        self.viewer = napari.view_image(self.data, ndisplay=2)

    def teardown(self, n):
        self.viewer.close()

    def time_change_contrast(self, n):
        """Time to change contrast limits."""
        self.viewer.layers[0].contrast_limits = (250, 3000)
        self.viewer.layers[0].contrast_limits = (300, 2900)
        self.viewer.layers[0].contrast_limits = (350, 2800)

    def time_change_gamma(self, n):
        """Time to change gamma."""
        self.viewer.layers[0].gamma = 0.5
        self.viewer.layers[0].gamma = 0.8
        self.viewer.layers[0].gamma = 1.3


class QtVolumeRenderingSuite:
    """Benchmarks for a single image layer in the viewer."""

    params = [2 ** i for i in range(4, 10)]

    def setup(self, n):
        _ = QApplication.instance() or QApplication([])
        np.random.seed(0)
        self.data = np.random.random((n, n, n)) * 2 ** 12
        self.viewer = napari.view_image(self.data, ndisplay=3)

    def teardown(self, n):
        self.viewer.close()

    def time_change_contrast(self, n):
        """Time to change contrast limits."""
        self.viewer.layers[0].contrast_limits = (250, 3000)
        self.viewer.layers[0].contrast_limits = (300, 2900)
        self.viewer.layers[0].contrast_limits = (350, 2800)

    def time_change_gamma(self, n):
        """Time to change gamma."""
        self.viewer.layers[0].gamma = 0.5
        self.viewer.layers[0].gamma = 0.8
        self.viewer.layers[0].gamma = 1.3
