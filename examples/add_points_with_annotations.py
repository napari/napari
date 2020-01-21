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
    size = np.array([10, 20, 20])
    annotations = {
        'point_class': np.array(['A', 'A', 'B']),
        'best_point': np.array([True, False, False])
    }
    face_color_cycle = ['blue', 'green']
    points_layer = viewer.add_points(
        points,
        size=size,
        annotations=annotations,
        face_color='point_class',
        face_color_cycle=face_color_cycle
    )
    #points_layer.face_color = 'point_class'
