# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/main/docs/BENCHMARKS.md

# TODO: (before PR merge)
#   Adjust data sizes to represent real sample data and access patterns


import time

import numpy as np
import zarr
from qtpy.QtWidgets import QApplication

import napari
from napari.layers import Image


class SlowMemoryStore(zarr.storage.MemoryStore):
    def __init__(self, load_delay, *args, **kwargs):
        self.load_delay = load_delay
        super().__init__(*args, **kwargs)

    def __getitem__(self, item: str):
        time.sleep(self.load_delay)
        return super().__getitem__(item)


class AsyncImage2DSuite:
    chunksize = [256, 512, 1024, 2048]
    latency = [0.05 * i for i in range(0, 4)]
    params = (latency, chunksize)

    def setup(self, latency, chunksize):
        store = SlowMemoryStore(load_delay=latency)
        self.data = zarr.zeros(
            (64, 2048, 2048),
            chunks=(1, chunksize, chunksize),
            dtype='uint8',
            store=store,
        )
        self.layer = Image(self.data)

    def time_create_layer(self, *args):
        """Time to create an image layer."""
        Image(self.data)

    def time_set_view_slice(self, *args):
        """Time to set view slice."""
        self.layer._set_view_slice()

    def time_refresh(self, *args):
        """Time to refresh view."""
        self.layer.refresh()


class QtViewerAsyncImage2DSuite:
    chunksize = [256, 512, 1024, 2048]
    latency = [0.05 * i for i in range(0, 4)]
    params = (latency, chunksize)

    def setup(self, latency, chunksize):
        store = SlowMemoryStore(load_delay=latency)
        _ = QApplication.instance() or QApplication([])

        self.data = zarr.zeros(
            (64, 2048, 2048),
            chunks=(1, chunksize, chunksize),
            dtype='uint8',
            store=store,
        )

        self.viewer = napari.Viewer()
        self.viewer.add_image(self.data)

    def time_z_scroll(self, *args):
        for z in range(self.data.shape[0]):
            self.viewer.dims.set_current_step(0, z)

    def teardown(self, *args):
        self.viewer.window.close()


class QtViewerAsyncPointsSuite:
    n_points = [2**i for i in range(12, 18)]
    params = n_points

    def setup(self, n_points):
        _ = QApplication.instance() or QApplication([])

        np.random.seed(0)
        self.viewer = napari.Viewer()
        # Fake image layer to set bounds. Is this really needed?
        self.empty_image = np.zeros((512, 512, 512), dtype="uint8")
        self.viewer.add_image(self.empty_image)
        self.point_data = np.random.randint(512, size=(n_points, 3))
        self.viewer.add_points(self.point_data)

    def time_z_scroll(self, *args):
        for z in range(self.empty_image.shape[0]):
            self.viewer.dims.set_current_step(0, z)

    def teardown(self, *args):
        self.viewer.window.close()


class QtViewerAsyncPointsAndImage2DSuite:
    n_points = [2**i for i in range(12, 18, 2)]
    chunksize = [256, 512, 1024]
    latency = [0.05 * i for i in range(0, 3)]
    params = (n_points, latency, chunksize)

    def setup(self, n_points, latency, chunksize):
        store = SlowMemoryStore(load_delay=latency)
        _ = QApplication.instance() or QApplication([])

        np.random.seed(0)

        self.image_data = zarr.zeros(
            (64, 2048, 2048),
            chunks=(1, chunksize, chunksize),
            dtype='uint8',
            store=store,
        )

        self.viewer = napari.Viewer()
        self.viewer.add_image(self.image_data)
        self.point_data = np.random.randint(512, size=(n_points, 3))
        self.viewer.add_points(self.point_data)

    def time_z_scroll(self, *args):
        for z in range(self.image_data.shape[0]):
            self.viewer.dims.set_current_step(0, z)

    def teardown(self, *args):
        self.viewer.window.close()
