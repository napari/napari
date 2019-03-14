"""
Display one markers layer ontop of one image layer using the add_markers and
add_image APIs
"""

import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage.segmentation import slic
from napari import Window, Viewer
from napari.util import app_context


with app_context():
    viewer = Viewer()
    window = Window(viewer)
    # add the image
    astro = data.astronaut()
    viewer.add_image(astro, multichannel=True)
    # add the labels
    labels = slic(astro, multichannel=True)
    label_layer = viewer.add_labels(labels)

