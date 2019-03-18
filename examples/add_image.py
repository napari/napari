"""
Displays one image using the add_image API and then adjust some of its
properties
"""

from skimage import data
from skimage.color import rgb2gray
from napari import ViewerApp
from napari.util import app_context


with app_context():
    # create the viewer with an image
    viewer = ViewerApp(astronaut=rgb2gray(data.astronaut()))

    # adjust some of the layer properties
    layer = viewer.layers[0]

    # change the layer name
    layer.name = 'astronaut'

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

    # change the layer colormap and color limits
    layer.colormap = 'gray'
    layer.clim = (0.1, 0.9)

    # change the layer interpolation mode
    layer.interpolation = 'bicubic'
    layer.interpolation = 'nearest'
