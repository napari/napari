"""
Display one markers layer ontop of one image layer using the add_markers and
add_image APIs
"""

import numpy as np
from skimage import data
from skimage.color import rgb2gray
from napari import ViewerApp
from napari.util import app_context


with app_context():
    # create the viewer and window
    viewer = ViewerApp()

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

    # set the layer property widget to be expanded
    layer._qt_properties.setExpanded(True)

    # change the layer opacity
    layer.opacity = 0.9

    # change the layer blending mode
    layer.blending = 'opaque'
    layer.blending = 'translucent'

    # change the layer marker face color
    layer.face_color = 'white'

    # change the layer marker edge color
    layer.edge_color = 'blue'

    # change the layer marker symbol
    layer.symbol = 'cross'

    # change the layer marker n_dimensional status
    layer.n_dimensional = True

    # change the layer marker size
    layer.size = 20

    # change the layer mode
    layer.mode = 'add'
