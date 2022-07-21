"""
Add points with multicolor text
===============================

Display a points layer on top of an image layer with text using
multiple face colors mapped from features for the points and text.
"""

import numpy as np
import napari

# add the image with three points
viewer = napari.view_image(np.zeros((400, 400)))
points = np.array([[100, 100], [200, 300], [333, 111]])

# create features for each point
features = {
    'confidence': np.array([1, 0.5, 0]),
    'good_point': np.array([True, False, False]),
}

# define the color cycle for the points face and text colors
color_cycle = ['blue', 'green']

text = {
    'string': 'Confidence is {confidence:.2f}',
    'size': 20,
    'color': {'feature': 'good_point', 'colormap': color_cycle},
    'translation': np.array([-30, 0]),
}

# create a points layer where the face_color is set by the good_point feature
# and the edge_color is set via a color map (grayscale) on the confidence
# feature
points_layer = viewer.add_points(
    points,
    features=features,
    text=text,
    size=20,
    edge_width=7,
    edge_width_is_relative=False,
    edge_color='confidence',
    edge_colormap='gray',
    face_color='good_point',
    face_color_cycle=color_cycle,
)

# set the edge_color mode to colormap
points_layer.edge_color_mode = 'colormap'

if __name__ == '__main__':
    napari.run()
