# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/main/docs/BENCHMARKS.md
import numpy as np
from qtpy.QtWidgets import QApplication

import napari

class QtViewerViewVectorSuite:
    """Benchmarks for viewing vectors in the viewer."""

    params = [2**i for i in range(4, 18, 2)]

    def setup(self, n):
        _ = QApplication.instance() or QApplication([])
        np.random.seed(0)
        self.data = np.random.random((n, 2, 3))
        self.viewer = napari.Viewer()
        self.layer = self.viewer.add_vectors(self.data)
        self.visual = self.viewer.window._qt_viewer.layer_to_visual[self.layer]

    def teardown(self, n):
        self.viewer.window.close()

    def time_vectors_refresh(self, n):
        """Time to view a vector."""
        self.viewer.layers[0].refresh()
        self.viewer.layers[0].refresh()
        self.viewer.layers[0].refresh()
