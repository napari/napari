"""
Cursor position
===============

Add small data to examine cursor positions

.. tags:: interactivity
"""

import numpy as np

import napari

viewer = napari.Viewer()
image = np.array([[1, 0, 0, 1],
                  [0, 0, 1, 1],
                  [1, 0, 3, 0],
                  [0, 2, 0, 0]], dtype=int)

viewer.add_labels(image)

points = np.array([[0, 0], [2, 0], [1, 3]])
viewer.add_points(points, size=0.25)

rect = np.array([[0, 0], [3, 1]])
viewer.add_shapes(rect, shape_type='rectangle', edge_width=0.1)

vect = np.array([[[3, 2], [-1, 1]]])
viewer.add_vectors(vect, edge_width=0.1)

if __name__ == '__main__':
    napari.run()
