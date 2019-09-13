"""
Display one shapes layer ontop of one image layer using the add_shapes and
add_image APIs. When the window is closed it will print the coordinates of
your shapes.
"""

import numpy as np
from skimage import data
import napari


with napari.gui_qt():
    # create the viewer and window
    viewer = napari.Viewer(ndisplay=3)

    # add the image
    #layer = viewer.add_image(data.camera(), name='photographer')

    # create a list of polygons
    polygons = [
        np.array([[11, 13, 1], [111, 113, 1], [22, 246, 2]]),
        np.array(
            [
                [505, 60, 1],
                [402, 71, 2],
                [383, 42, 1],
                [251, 95, 2],
                [212, 59, 1],
                [131, 137, 2],
                [126, 187, 1],
                [191, 204, 2],
                [171, 248, 1],
                [211, 260, 2],
                [273, 243, 1],
                [264, 225, 2],
                [430, 173, 1],
                [512, 160, 2],
            ]
        ),
        np.array(
            [
                [310, 382, 1],
                [229, 381, 2],
                [209, 401, 2],
                [221, 411, 2],
                [258, 411, 1],
                [300, 412, 2],
                [306, 435, 2],
                [268, 434, 2],
                [265, 454, 2],
                [298, 461, 2],
                [307, 461, 1],
                [307, 507, 2],
                [349, 510, 1],
                [352, 369, 2],
                [330, 366, 2],
                [330, 366, 2],
            ]
        ),
    ]

    # add polygons
    layer = viewer.add_shapes(
        polygons,
        shape_type='path',
        edge_width=1,
        edge_color='coral',
        face_color='royalblue',
        name='shapes',
    )

    # change some properties of the layer
    layer.selected_data = list(range(layer.nshapes))
    layer.edge_width = 5
    layer.opacity = 0.75
    layer.selected_data = []

    # add an ellipse to the layer
    ellipse = np.array([[59, 222], [110, 289], [170, 243], [119, 176]])
    layer.add(
        ellipse,
        shape_type='ellipse',
        edge_width=5,
        edge_color='coral',
        face_color='purple',
        opacity=0.75,
    )

    # Set the layer mode with a string
    layer.mode = 'select'

# Print the shape coordinate data
print(layer.nshapes, "shapes at:")
print(layer.data)
