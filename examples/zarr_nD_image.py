"""
Display one 4-D image layer using the add_image API
"""

import dask.array as da
import zarr
import numpy as np
from skimage import data
from napari import ViewerApp
from napari.util import app_context


with app_context():
    data = zarr.zeros((102_000, 200, 210), chunks=(100, 200, 210))
    data[53_000:53_100, 100:110, 110:120] = 1

    array = da.from_zarr(data)
    print(array.shape)
    viewer = ViewerApp(array, clim_range=[0, 1], multichannel=False)
