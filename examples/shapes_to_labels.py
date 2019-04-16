"""
Display one shapes layer ontop of one image layer using the add_shapes and
add_image APIs. When the window is closed it will print the coordinates of
your shapes.
"""

import numpy as np
from skimage import data
from skimage.color import rgb2gray
from napari import ViewerApp
from napari.util import app_context
from vispy.color import Colormap

with app_context():
    # create the viewer and window
    viewer = ViewerApp()

    # add the image
    layer = viewer.add_image(data.camera(), name='photographer')
    layer.colormap = 'gray'

    # create a list of polygons
    polygons = ([np.array([[13,  11], [113, 111], [246, 22]]),
                 np.array([[60, 505], [71, 402], [42, 383], [95, 251],
                           [59, 212], [137, 131], [187, 126], [204, 191],
                           [248, 171], [260, 211], [243, 273], [225, 264],
                           [173, 430], [160, 512]]),
                 np.array([[382, 310], [381, 229], [401, 209], [411, 221],
                           [411, 258], [412, 300], [435, 306], [434, 268],
                           [454, 265], [461, 298], [461, 307], [507, 307],
                           [510, 349], [369, 352], [366, 330], [366, 330]])])

    # add polygons
    layer = viewer.add_shapes(polygons, shape_type='polygon', edge_width=1,
                              edge_color='coral', face_color='royalblue',
                              name='shapes')

    # change some properties of the layer
    layer.selected_shapes = list(range(len(layer.data.shapes)))
    layer.edge_width = 5
    layer.opacity = 0.75
    layer.selected_shapes = []

    # add an ellipse to the layer
    ellipse = np.array([[222, 59], [289, 110], [243, 170], [176, 119]])
    layer.add_shapes(ellipse, shape_type='ellipse', edge_width=5,
                     edge_color='coral', face_color='purple',
                     opacity=0.75)
    layer.refresh()

    layer._qt_properties.setExpanded(True)

    masks = layer.data.to_masks([512, 512]).transpose(2, 1, 0)
    masks_layer = viewer.add_image(masks.astype(float), multichannel=False,
                                   name='masks')
    masks_layer.opacity = 0.7
    masks_layer.colormap = Colormap([[0.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 1.0]])

    labels = layer.data.to_labels([512, 512]).transpose(1, 0)
    labels_layer = viewer.add_labels(labels, name='labels')
    labels_layer.visible = False
