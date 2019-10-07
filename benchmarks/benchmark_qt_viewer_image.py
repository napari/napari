# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/master/BENCHMARKS.md
import numpy as np
import napari
from qtpy.QtWidgets import QApplication


class QtViewerViewImageSuite:
    """Benchmarks for viewing images in the viewer."""

    params = [2 ** i for i in range(4, 13)]

    def setup(self, n):
        app = QApplication.instance() or QApplication([])
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
        app = QApplication.instance() or QApplication([])
        np.random.seed(0)
        self.data = np.random.random((n, n))
        self.viewer = napari.Viewer()

    def teardown(self, n):
        self.viewer.window.close()

    def time_add_image(self, n):
        """Time to view an image."""
        self.viewer.add_image(self.data)


class QtViewerEditImageSuite:
    """Benchmarks for editing images in the viewer."""

    params = [2 ** i for i in range(4, 13)]

    def setup(self, n):
        app = QApplication.instance() or QApplication([])
        np.random.seed(0)
        self.data = np.random.random((n, n))
        self.viewer = napari.view_image(self.data)

    def teardown(self, n):
        self.viewer.window.close()

    def time_set_view_slice(self, n):
        """Time to set view slice."""
        self.viewer.layers[0]._set_view_slice()

    def time_update_thumbnail(self, n):
        """Time to update thumbnail."""
        self.viewer.layers[0]._update_thumbnail()

    def time_get_value(self, n):
        """Time to get current value."""
        self.viewer.layers[0].get_value()
