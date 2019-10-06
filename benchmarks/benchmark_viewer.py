# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/master/BENCHMARKS.md
import numpy as np
import napari


class ViewerSuite:
    """Benchmarks for the Viewer."""

    def setup(self):
        shape = (10, 10)
        np.random.seed(0)
        self.data = np.random.random(shape)

    def time_empty_viewer(self):
        """Time to create an empty viewer."""
        with napari.gui_qt:
            napari.Viewer()

    def time_view_image(self):
        """Time to view an image."""
        with napari.gui_qt:
            napari.view_image(self.data)
