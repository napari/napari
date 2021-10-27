"""
Display a zarr array
"""

try:
    import zarr
except ImportError:
    raise ImportError("""This example uses a zarr array but zarr is not
    installed. To install try 'pip install zarr'.""")

import napari


data = zarr.zeros((102_0, 200, 210), chunks=(100, 200, 210))
data[53_0:53_1, 100:110, 110:120] = 1

print(data.shape)
# For big data, we should specify the contrast_limits range, or napari will try
# to find the min and max of the full image.
viewer = napari.view_image(data, contrast_limits=[0, 1], rgb=False)

napari.run()
