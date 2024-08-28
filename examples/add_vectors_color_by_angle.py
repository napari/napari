"""
Add vectors color by angle
==========================

This example generates a set of vectors in a spiral pattern.
The color of the vectors is mapped to their 'angle' feature.

.. tags:: visualization-advanced
"""

import numpy as np
from skimage import data

import napari

# create the viewer and window
viewer = napari.Viewer()

layer = viewer.add_image(data.camera(), name='photographer')

# sample vector coord-like data
n = 300
pos = np.zeros((n, 2, 2), dtype=np.float32)
phi_space = np.linspace(0, 4 * np.pi, n)
radius_space = np.linspace(0, 100, n)

# assign x-y position
pos[:, 0, 0] = radius_space * np.cos(phi_space) + 300
pos[:, 0, 1] = radius_space * np.sin(phi_space) + 256

# assign x-y projection
pos[:, 1, 0] = 2 * radius_space * np.cos(phi_space)
pos[:, 1, 1] = 2 * radius_space * np.sin(phi_space)

# make the angle feature, range 0-2pi
angle = np.mod(phi_space, 2 * np.pi)

# create a feature that is true for all angles  > pi
pos_angle = angle > np.pi

# create the features dictionary.
features = {
    'angle': angle,
    'pos_angle': pos_angle,
}

# define the edge color as part of the overall visualization style
style = {
    'edge_color': {'feature': 'angle', 'colormap': 'husl'}
}

# add the vectors
layer = viewer.add_vectors(
    pos,
    edge_width=3,
    features=features,
    style=style,
    name='vectors'
)

if __name__ == '__main__':
    napari.run()
