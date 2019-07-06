"""
Displays one image using the add_image API and then adjust some of its
properties
"""

from skimage import data
from skimage.color import rgb2gray
import napari


with napari.gui_qt():
    # create the viewer with an image
    viewer = napari.view(
        astronaut=rgb2gray(data.astronaut()), title='napari example'
    )

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

    # change the layer opacity
    layer.opacity = 0.9

    # change the layer blending mode
    layer.blending = 'opaque'
    layer.blending = 'translucent'

    # change the layer colormap and color limits
    layer.colormap = 'gray'
    layer.clim = (0, 0.9)

    # change the layer interpolation mode
    layer.interpolation = 'bicubic'
    layer.interpolation = 'nearest'
