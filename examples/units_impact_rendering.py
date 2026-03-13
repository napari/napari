"""
Layer units impact rendering
============================

This example demonstrates how units can impact rendering in napari.
Two layers are added with different scales, however by specifying the units,
napari is able to render them in the same physical space by the unit-aware
library Pint. The first layer is in nanometers and the second layer is
in micrometers, but they are rendered in the same physical space because 1 μm = 1000 nm.

.. tags:: visualization-advanced, layers
"""

import skimage

import napari

data = skimage.data.cells3d()

ch1 = data[:, 0]
ch2 = data[:, 1]

viewer = napari.Viewer()
viewer.add_image(ch1, units=('nm', 'nm', 'nm'), name='ch1', scale=(210, 70, 70), colormap="magenta")
viewer.add_image(ch2, units=('μm', 'μm', 'μm'), name='ch2', scale=(0.210, 0.07, 0.07), colormap="green", blending='additive')

if __name__ == '__main__':
    napari.run()
