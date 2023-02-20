"""
Spherical points
================

.. tags:: experimental
"""
import numpy as np

import napari

np.random.seed()

pts = np.random.rand(100, 3) * 100
colors = np.random.rand(100, 3)
sizes = np.random.rand(100) * 20 + 10

viewer = napari.Viewer(ndisplay=3)
pts_layer = viewer.add_points(
    pts,
    face_color=colors,
    size=sizes,
    shading='spherical',
    edge_width=0,
)

# antialiasing is currently a bit broken, this is especially bad in 3D so
# we turn it off here
pts_layer.antialiasing = 0

viewer.reset_view()

if __name__ == '__main__':
    napari.run()
