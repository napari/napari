"""
Display a points layer on top of an image layer using the add_points and
add_image APIs
"""

import numpy as np
from skimage import data
from skimage.color import rgb2gray
import napari


with napari.gui_qt():
    # create the viewer window
    viewer = napari.Viewer()

    # add the image
    viewer.add_image(np.random.random((4, 4, 104, 139)))
    # add the points

    viewer.add_image(np.random.random((5, 10, 10, 104, 139)))
    # points = np.round(4 * np.random.random((60272, 4))).astype(int)
    # viewer.add_points(points)
