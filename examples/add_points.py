"""
Add points
==========

Display a points layer on top of an image layer using the ``add_points`` and
``add_image`` APIs

.. tags:: visualization-basic
"""

import numpy as np
from skimage import data
from skimage.color import rgb2gray
import napari


# add the image
viewer = napari.view_image(rgb2gray(data.astronaut()))
# add the points
points = np.array([[100, 100], [200, 200], [333, 111]])
size = np.array([10, 20, 20])
viewer.add_points(points, size=size)

# unselect the image layer
viewer.layers.selection.discard(viewer.layers[0])

# adjust some of the points layer attributes
layer = viewer.layers[1]

# change the layer name
layer.name = 'points'

# change the layer visibility
layer.visible = False
layer.visible = True

# select the layer
viewer.layers.selection.add(layer)
# deselect the layer
viewer.layers.selection.remove(layer)
# or: viewer.layers.selection.discard(layer)

# change the layer opacity
layer.opacity = 0.9

# change the layer point symbol using an alias
layer.symbol = '+'

# change the layer point out_of_slice_display status
layer.out_of_slice_display = True

# change the layer mode
layer.mode = 'add'

if __name__ == '__main__':
    napari.run()
