"""
Loads and Displays tiffs as they get generated in the specific directory.
Trying to simulate the live display of data as it gets acquired by microscope. 
"""

import os
import sys
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
    # pass a directory to monitor or it will monitor current directory.
    path = sys.argv[1] if len(sys.argv) > 1 else '.' 
    path = os.path.abspath(path)
    end_of_experiment = 'final.log'

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

        if viewer.dims.point[0] >= layer.data.shape[0] - 2:
            viewer.dims.set_point(0, layer.data.shape[0] - 1)
   
  
    @thread_worker(connect={'yielded': append})
    def watch_path(path):
        current_files = set()
        processed_files = set()
        while True:
            files_to_process = set()
            current_files = set(os.listdir(path))
            if end_of_experiment in current_files:
                files_to_process = current_files - processed_files
                files_to_process.remove(end_of_experiment)

            elif len(current_files):
                last_file = sorted(current_files, key=alphanumeric_key)[-1]
                current_files.remove(last_file)
                files_to_process = current_files - processed_files
            
            for p in sorted(files_to_process, key=alphanumeric_key):
                yield delayed(imread)(os.path.join(path, p))
            else:
                yield

            processed_files.update(files_to_process)
            time.sleep(0.1)

    worker = watch_path(path)
    