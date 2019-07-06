"""
Display one 5-D image layer using the add_image API
"""

import numpy as np
from skimage import data
import napari


with napari.gui_qt():

    viewer = napari.view(np.random.random((10, 20, 15, 30, 40)))
