"""
Xarray example
==============

Displays an xarray

.. tags:: visualization-nD
"""

try:
    import xarray as xr
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        """This example uses a xarray but xarray is not
    installed. To install try 'pip install xarray'."""
    ) from None

import numpy as np

import napari

data = np.random.random((20, 40, 50))
xdata = xr.DataArray(data, dims=['z', 'y', 'x'])

# create an empty viewer
viewer = napari.Viewer()

# add the xarray
layer = viewer.add_image(xdata, name='xarray')

if __name__ == '__main__':
    napari.run()
