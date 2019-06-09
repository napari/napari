"""
Display a markers layer on top of an image layer using the add_markers and
add_image APIs
"""

import numpy as np
from skimage import data
from skimage.color import rgb2gray
import napari
from napari.util import app_context


with app_context():
    # create the viewer window
    viewer = napari.Viewer()

    # add the image
    viewer.add_image(rgb2gray(data.astronaut()))
    # add the markers
    markers = np.array([[100, 100], [200, 200], [333, 111]])
    size = np.array([10, 20, 20])
    viewer.add_markers(markers, size=size)

    # unselect the image layer
    viewer.layers[0].selected = False

    # adjust some of the marker layer properties
    layer = viewer.layers[1]

    # change the layer name
    layer.name = 'spots'

    # change the layer visibility
    layer.visible = False
    layer.visible = True

    # change the layer selection
    layer.selected = False
    layer.selected = True

    # change the layer opacity
    layer.opacity = 0.9

    # change the layer blending mode
    layer.blending = 'opaque'
    layer.blending = 'translucent'

    # change the layer marker face color
    layer.face_color = 'white'

    # change the layer marker edge color
    layer.edge_color = 'blue'

    # change the layer marker symbol using an alias
    layer.symbol = '+'

    # change the layer marker n_dimensional status
    layer.n_dimensional = True

    # change the layer marker size
    layer.size = 20
    layer.size = np.array([10, 50, 20])

    # change the layer mode
    layer.mode = 'add'
