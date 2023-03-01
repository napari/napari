"""
Annotate 2D
===========

Display one points layer ontop of one image layer using the ``add_points`` and
``add_image`` APIs

.. tags:: analysis
"""

import numpy as np
from skimage import data

import napari

print("click to add points; close the window when finished.")

viewer = napari.view_image(data.astronaut(), rgb=True)
points = viewer.add_points(np.zeros((0, 2)))
points.mode = 'add'

if __name__ == '__main__':
    napari.run()

    print("you clicked on:")
    print(points.data)
