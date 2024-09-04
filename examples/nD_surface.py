"""
nD surface
==========

Display a 3D surface

.. tags:: visualization-nD
"""

import numpy as np

import napari

# create the viewer and window
viewer = napari.Viewer(ndisplay=3)

data = np.array([[0, 0, 0], [0, 20, 10], [10, 0, -10], [10, 10, -10]])
faces = np.array([[0, 1, 2], [1, 2, 3]])
values = np.linspace(0, 1, len(data))

# add the surface
layer = viewer.add_surface((data, faces, values))

if __name__ == '__main__':
    napari.run()
