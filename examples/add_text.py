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
    
    annotations = ['hi', 'hola', 'bonjour']
    font_size = (10, 20, 30)


    viewer.add_text(points, text=annotations, font_size=font_size)

    viewer.layers[1].text_color = 'green'
    viewer.layers[1].font_size = 20
