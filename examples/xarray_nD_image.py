"""
Displays an xarray
"""

try:
    import xarray as xr
except ImportError:
    raise ImportError("""This example uses a xarray but xarray is not
    installed. To install try 'pip install xarray'.""")

import numpy as np
import napari

data = np.random.random((20, 40, 50))
xdata = xr.DataArray(data, dims=['z', 'y', 'x'])

# create an empty viewer
viewer = napari.Viewer()

# add the xarray
layer = viewer.add_image(xdata, name='xarray')

napari.run()
