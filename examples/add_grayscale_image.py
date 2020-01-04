"""
Display one image using the add_image API.
"""

from skimage import data
import napari
import numpy as np


with napari.gui_qt():
    # create the viewer with an image
    viewer = napari.view_image(data.astronaut().mean(-1).astype(np.uint16))
