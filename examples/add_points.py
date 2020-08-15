"""
Display a points layer on top of an image layer using the add_points and
add_image APIs
"""

import numpy as np
from skimage import data
from skimage.color import rgb2gray
import napari


with napari.gui_qt():
    # add the image
    viewer = napari.view_image(rgb2gray(data.astronaut()))
    # add the points
    points = np.array([[100, 100], [200, 200], [333, 111]])
    size = np.array([10, 20, 20])
    viewer.add_points(points, size=size)

    # unselect the image layer
    viewer.layers[0].selected = False

    # adjust some of the points layer properties
    layer = viewer.layers[1]

    # change the layer name
    layer.name = 'points'

    # change the layer visibility
    layer.visible = False
    layer.visible = True

    # change the layer selection
    layer.selected = False
    layer.selected = True

    # change the layer opacity
    layer.opacity = 0.9

    # change the layer point symbol using an alias
    layer.symbol = '+'

    # change the layer point n_dimensional status
    layer.n_dimensional = True

    # change the layer mode
    layer.mode = 'add'

