"""
Update console
==============

Display one shapes layer ontop of one image layer using the add_shapes and
add_image APIs. When the window is closed it will print the coordinates of
your shapes.

.. tags:: historical
"""

import numpy as np
from skimage import data

import napari

# create the viewer and window
viewer = napari.Viewer()

# add the image
photographer = data.camera()
image_layer = viewer.add_image(photographer, name='photographer')

# create a list of polygons
polygons = [
    np.array([[11, 13], [111, 113], [22, 246]]),
    np.array(
        [
            [505, 60],
            [402, 71],
            [383, 42],
            [251, 95],
            [212, 59],
            [131, 137],
            [126, 187],
            [191, 204],
            [171, 248],
            [211, 260],
            [273, 243],
            [264, 225],
            [430, 173],
            [512, 160],
        ]
    ),
    np.array(
        [
            [310, 382],
            [229, 381],
            [209, 401],
            [221, 411],
            [258, 411],
            [300, 412],
            [306, 435],
            [268, 434],
            [265, 454],
            [298, 461],
            [307, 461],
            [307, 507],
            [349, 510],
            [352, 369],
            [330, 366],
            [330, 366],
        ]
    ),
]

# add polygons
shapes_layer = viewer.add_shapes(
    polygons,
    shape_type='polygon',
    edge_width=5,
    edge_color='coral',
    face_color='royalblue',
    name='shapes',
)

# Send local variables to the console
viewer.update_console(locals())

if __name__ == '__main__':
    napari.run()
