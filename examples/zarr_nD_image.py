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
    data = zarr.zeros((200, 210, 102_000), chunks=(200, 210, 100))
    data[100:110, 110:120, 53_000:53_100] = 1

    array = da.from_zarr(data)
    viewer = ViewerApp(array)
