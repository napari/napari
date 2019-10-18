"""
Display a points layer on top of an image layer using the add_points and
add_image APIs
"""

import numpy as np
from skimage import data
from skimage.color import rgb2gray
import napari


with napari.gui_qt():
    # add the image
    viewer = napari.view_image(rgb2gray(data.astronaut()))
    # add the points
    points = np.array([[100, 100], [200, 200], [333, 111]])
    size = np.array([5, 5, 5])
    annotations = ['hi', 'hola', 'bonjour']
    annotation_offset = [0, 0]

    viewer.add_annotations(points, annotations=annotations, annotation_offset=annotation_offset, size=size)
