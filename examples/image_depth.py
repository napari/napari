"""
Image depth
===========

.. tags:: visualization-basic
"""

import numpy as np

import napari

im_data = np.zeros((50, 50, 50))
im_data[30:40, 25:35, 25:35] = 1
viewer = napari.view_image(im_data, colormap='magenta', rendering='iso')
viewer.add_image(im_data, colormap='green', rendering='iso', translate=(30, 0, 0))

points_data = [
    [50, 30, 30],
    [25, 30, 30],
    [75, 30, 30]
]
viewer.add_points(points_data, size=4)

viewer.dims.ndisplay = 3

if __name__ == '__main__':
    napari.run()
