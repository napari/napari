"""
Display one 4-D image layer using the add_image API
"""

import numpy as np
from skimage import data
from skimage.color import rgb2gray
from napari import Window, Viewer
from napari.util import app_context


with app_context():
    # create the viewer and window
    viewer = Viewer()
    window = Window(viewer)
    # add the image
    viewer.add_image(np.random.rand(500, 500, 20, 10))
