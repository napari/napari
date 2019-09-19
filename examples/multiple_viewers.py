"""
Create multiple viewers from the same script
"""

import numpy as np
from skimage import data
import napari


with napari.gui_qt():
    # add the image
    photographer = data.camera()
    viewer_a = napari.add_image(photographer, name='photographer')

    # add the image
    astronaut = data.astronaut()
    viewer_b = napari.add_image(astronaut, name='astronaut')
