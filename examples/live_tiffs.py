import os
import time
import numpy as np
from skimage.io.collection import alphanumeric_key
from dask import delayed
import dask.array as da
from tifffile import imread
import napari
from napari.qt import thread_worker


with napari.gui_qt():
    viewer = napari.Viewer(ndisplay=3)

    def append(delayed_image):
        if delayed_image is None:
            return

        if viewer.layers:
            layer = viewer.layers[0]
            image_shape = layer.data.shape[1:]
            image_dtype = layer.data.dtype
            dask_image = da.from_delayed(
                delayed_image,
                shape=image_shape,
                dtype=image_dtype,
            ).reshape((1,) + image_shape)
            layer.data = da.concatenate(
                (layer.data, dask_image), axis=0
            )
        else:
            image = delayed_image.compute()
            dask_image = da.from_delayed(
                delayed_image,
                shape=image.shape,
                dtype=image.dtype,
            ).reshape((1,) + image.shape)
            layer = viewer.add_image(dask_image, rendering='attenuated_mip')
        if viewer.dims.point[0] >= layer.data.shape[0] - 1:
            viewer.dims.set_point(0, layer.data.shape[0])

    @thread_worker(connect={'yielded': append})
    def watch_path(path):
        last_files = set()
        while True:
            now_files = set(os.listdir(path))
            for p in sorted(list(now_files - last_files), key=alphanumeric_key):
                yield delayed(imread)(os.path.join(path, p))
            else:
                yield
            last_files = now_files
            time.sleep(0.3)

    worker = watch_path("C:/Code/Lab/napari/examples/testop")