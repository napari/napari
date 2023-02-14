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


viewer = napari.Viewer()
viewer.theme = 'light'

# # add the image
# viewer = napari.view_image(rgb2gray(data.astronaut()))
# add the points
points = np.array([[100, 100], [200, 200], [333, 111]])
size = np.array([10, 20, 20])
# # viewer.add_points(points, size=size)
# viewer = napari.view_points(points, size=size)

# # adjust some of the points layer attributes
# layer = viewer.layers[0]

# # change the layer name
# layer.name = 'points'
# layer.visible = False

# from napari.settings import get_settings
# import napari

# settings = get_settings()
# # then modify... e.g:
# settings.appearance.theme = 'light'

layer = viewer.add_points(points, size=size)
layer.name = 'points2'

# change the layer opacity
layer.opacity = 0.9

# change the layer point symbol using an alias
layer.symbol = '+'

# # change the layer point out_of_slice_display status
# layer.out_of_slice_display = True

# # change the layer mode
# layer.mode = 'add'

if __name__ == '__main__':
    napari.run()



# pt = viewer.layers[0]