"""
Display a points layer on top of an image layer using the add_points and
add_image APIs
"""

import numpy as np
import napari


# add the image
viewer = napari.view_image(np.zeros((400, 400)))
# add the points
points = np.array([[100, 100], [200, 300], [333, 111]])

# create properties for each point
properties = {
    'confidence': np.array([1, 0.5, 0]),
    'good_point': np.array([True, False, False]),
}

# define the color cycle for the face_color annotation
face_color_cycle = ['blue', 'green']

text = {
    'text': 'Confidence is {confidence:.2f}',
    'size': 20,
    'color': 'green',
    'translation': np.array([-30, 0]),
}

# create a points layer where the face_color is set by the good_point property
# and the edge_color is set via a color map (grayscale) on the confidence property.
points_layer = viewer.add_points(
    points,
    properties=properties,
    text=text,
    size=20,
    edge_width=7,
    edge_color='confidence',
    edge_colormap='gray',
    face_color='good_point',
    face_color_cycle=face_color_cycle,
)

# set the edge_color mode to colormap
points_layer.edge_color_mode = 'colormap'

napari.run()
