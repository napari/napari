"""
Display one 5-D image layer using the add_image API
"""

import numpy as np
from skimage import data
import napari
from napari.util import app_context


with app_context():

    viewer = napari.view(np.random.random((10, 20, 15, 30, 40)))
