"""
Create multiple viewers from the same script
"""

import numpy as np
from skimage import data
import napari


with napari.gui_qt():
    # create the viewer and window
    viewer_a = napari.Viewer()

    # add the image
    photographer = data.camera()
    viewer_a.add_image(photographer, name='photographer')

    # create a new viewer
    viewer_b = napari.Viewer()

    # add the image
    astronaut = data.astronaut()
    viewer_b.add_image(astronaut, name='astronaut')
