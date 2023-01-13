"""
Add shapes with features
========================

Display one shapes layer ontop of one image layer using the ``add_shapes`` and
``add_image`` APIs. When the window is closed it will print the coordinates of
your shapes.

.. tags:: visualization-basic
"""

import numpy as np
from skimage import data
import napari


# add the image
viewer = napari.view_image(data.camera(), name='photographer')

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

# create features
features = {
    'likelihood': [0.2, 0.5, 1],
    'class': ['sky', 'person', 'building'],
}
face_color_cycle = ['blue', 'magenta', 'green']

# add polygons
layer = viewer.add_shapes(
    polygons,
    features=features,
    shape_type='polygon',
    edge_width=1,
    edge_color='likelihood',
    edge_colormap='gray',
    face_color='class',
    face_color_cycle=face_color_cycle,
    name='shapes',
)

# change some attributes of the layer
layer.selected_data = set(range(layer.nshapes))
layer.current_edge_width = 5
layer.selected_data = set()

# To save layers to svg:
# viewer.layers.save('viewer.svg', plugin='svg')

if __name__ == '__main__':
    napari.run()
