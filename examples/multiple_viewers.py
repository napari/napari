"""
Create multiple viewers from the same script
"""

import numpy as np
from skimage import data
import napari


with napari.gui_qt():
    # add the image
    photographer = data.camera()
    viewer_a = napari.view_image(photographer, name='photographer')

    # add the image in a new viewer window
    astronaut = data.astronaut()
    # Also view_path, view_shapes, view_points, view_labels etc.
    viewer_b = napari.view_image(astronaut, name='astronaut')
