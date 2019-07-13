"""
Displays an xarray
"""

import numpy as np
import xarray as xr
from napari import Viewer, gui_qt

data = np.random.random((20, 40, 50))
xdata = xr.DataArray(data, dims=['z', 'y', 'x'])

with gui_qt():
    # create an empty viewer
    viewer = Viewer()

    # add the xarray
    layer = viewer.add_image(xdata, name='xarray')
