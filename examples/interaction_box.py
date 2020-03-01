"""
Demonstrate interaction box on image layer
"""

from skimage import data
import napari
import numpy as np


with napari.gui_qt():
    # create the viewer with an image
    viewer = napari.view_image(data.astronaut(), rgb=True)
    viewer.active_layer.interactive = False
    viewer.active_layer._interaction_box.points = np.array([[50,50],[300,302]])
    viewer.active_layer._interaction_box.show = True
