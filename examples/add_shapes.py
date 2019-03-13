"""
Display one markers layer ontop of one image layer using the add_markers and
add_image APIs
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
    viewer.add_image(data.camera())
    viewer.layers[0].visible = False

    # add the shapes
    ellipses = np.array([[[195, 180], [ 45, 130]], [[295,  80], [395, 380]], [[ 95,
                        80], [105,  90]]])
    points = np.array([[100, 100], [200, 200], [333, 111]])
    polygons = [points+[-22,101], points-[140, -33], points+[19,-21]]
    viewer.add_shapes(polygons=polygons)
    viewer.layers[-1].edge_width = 10
    viewer.layers[-1].edge_color = 'coral'
