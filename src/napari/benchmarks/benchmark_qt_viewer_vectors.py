# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/main/docs/BENCHMARKS.md
import os

import numpy as np
from packaging.version import parse as parse_version
from qtpy.QtWidgets import QApplication

import napari

NAPARI_0_4_19 = parse_version(napari.__version__) <= parse_version('0.4.19')


class QtViewerViewVectorSuite:
    """Benchmarks for viewing vectors in the viewer."""

    params = [2**i for i in range(4, 18, 2)]

    if 'PR' in os.environ:
        skip_params = [(2**i,) for i in range(8, 18, 2)]

    def setup(self, n):
        _ = QApplication.instance() or QApplication([])
        np.random.seed(0)
        self.data = np.random.random((n, 2, 3))
        self.viewer = napari.Viewer()
        self.layer = self.viewer.add_vectors(self.data)
        if NAPARI_0_4_19:
            self.visual = self.viewer.window._qt_viewer.layer_to_visual[
                self.layer
            ]
        else:
            self.visual = self.viewer.window._qt_viewer.canvas.layer_to_visual[
                self.layer
            ]

    def teardown(self, n):
        self.viewer.window.close()

    def time_vectors_refresh(self, n):
        """Time to refresh a vector."""
        self.viewer.layers[0].refresh()

    def time_vectors_multi_refresh(self, n):
        """Time to refresh a vector multiple times."""
        self.viewer.layers[0].refresh()
        self.viewer.layers[0].refresh()
        self.viewer.layers[0].refresh()


if __name__ == '__main__':
    from utils import run_benchmark

    run_benchmark()
