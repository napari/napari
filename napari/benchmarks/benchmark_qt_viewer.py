# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/main/docs/BENCHMARKS.md
import napari


class QtViewerSuite:
    """Benchmarks for viewing images in the viewer."""

    def setup(self):
        self.viewer = None

    def teardown(self):
        self.viewer.window.close()

    def time_create_viewer(self):
        """Time to create the viewer."""
        self.viewer = napari.Viewer()
