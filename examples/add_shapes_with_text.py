"""
Add shapes with text
====================

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
    np.array([[225, 146], [283, 146], [283, 211], [225, 211]]),
    np.array([[67, 182], [167, 182], [167, 268], [67, 268]]),
    np.array([[111, 336], [220, 336], [220, 240], [111, 240]]),
]

# create features
features = {
    'likelihood': [21.23423, 51.2315, 100],
    'class': ['hand', 'face', 'camera'],
}
edge_color_cycle = ['blue', 'magenta', 'green']

text = {
    'string': '{class}: {likelihood:0.1f}%',
    'anchor': 'upper_left',
    'translation': [-5, 0],
    'size': 8,
    'color': 'green',
}

# add polygons
shapes_layer = viewer.add_shapes(
    polygons,
    features=features,
    shape_type='polygon',
    edge_width=3,
    edge_color='class',
    edge_color_cycle=edge_color_cycle,
    face_color='transparent',
    text=text,
    name='shapes',
)

# change some attributes of the layer
shapes_layer.opacity = 1

# To save layers to svg:
# viewer.layers.save('viewer.svg', plugin='svg')

if __name__ == '__main__':
    napari.run()
