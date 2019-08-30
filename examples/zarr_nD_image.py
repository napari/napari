"""
Display one 4-D image layer using the add_image API
"""

import dask.array as da
import zarr
import napari


with napari.gui_qt():
    data = zarr.zeros((102_000, 200, 210), chunks=(100, 200, 210))
    data[53_000:53_100, 100:110, 110:120] = 1

    array = da.from_zarr(data)
    print(array.shape)
    # For big data, we should specify the clim range, or napari will try
    # to find the min and max of the full image.
    viewer = napari.view(array, clim_range=[0, 1], multichannel=False)
