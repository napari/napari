"""
Displays an xarray
"""

import numpy as np
import xarray as xr
import napari

data = np.random.random((20, 40, 50))
xdata = xr.DataArray(data, dims=['z', 'y', 'x'])

with napari.gui_qt():
    # create an empty viewer
    viewer = napari.Viewer()

    # add the xarray
    layer = viewer.add_image(xdata, name='xarray')
