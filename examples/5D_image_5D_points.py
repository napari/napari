"""
Overlay 5D points on a 5D image
"""

import numpy as np
from skimage import data
from skimage.color import rgb2gray
import napari


with napari.gui_qt():
    # create the viewer window
    viewer = napari.Viewer()

    data = np.random.random((1, 1, 1, 100, 200))
    viewer.add_image(data)

    points = np.floor(5 * np.random.random((1000, 5))).astype(int)
    points[:, -2:] = 20 * points[:, -2:]
    viewer.add_points(points)
