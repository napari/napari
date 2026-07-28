"""
Displaying xarray data in napari
================================

This example shows how to view xarray datasets in napari.

napari automatically inherits axis_labels, scale, and units (if valid Pint
units), and translate from xarray DataArrays when available. 

Currently, napari cannot display irregularly-sampled data, so the code
assumes that the data indices are regularly spaced. If your indices are
irregular, use `xarray.Dataset.interp` to create a regularly-spaced version
before displaying it in napari.

.. tags:: visualization-advanced, layers, xarray
"""
import numpy as np
import xarray as xr

import napari

# open the xarray global sea surface temperature (40MB) and North America
# air temperature (30MB) datasets
sst = xr.tutorial.open_dataset('ersstv5')
airtemp = xr.tutorial.open_dataset('air_temperature')


# Show the raw (not resampled) model data
viewer, sst_layer = napari.imshow(
        sst.sst,
        name='sea surface temp',
        units=('ns', 'degrees', 'degrees'),
        colormap='magma',
        )

air_layer = viewer.add_image(
        airtemp.air,
        name='air temp NA',
        units=('ns', 'degrees', 'degrees'),
        colormap='viridis',
        blending='additive',
        contrast_limits=(-23 + 273, 32 + 273),  # data are in degrees Kelvin
        )


# set a time that overlaps both datasets
viewer.layers.units = ('hour', 'degrees', 'degrees')
viewer.dims.set_point(0, 383322)  # in hours

# latitude goes from -90 (south, down) to 90 (north, up),
# so we make sure that the camera vertical axis points up.
viewer.camera.orientation2d = ('up', 'right')
viewer.scale_bar.visible = True

# fill the frame
viewer.fit_to_view()


if __name__ == '__main__':
    napari.run()
