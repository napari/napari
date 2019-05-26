"""
Displays an image pyramid
"""

from skimage import data
from skimage.color import rgb2gray
from skimage.transform import pyramid_gaussian
import napari
from napari.util import app_context
import numpy as np


image = data.astronaut()
rows, cols, dim = image.shape


# create pyramid from astronaut image
astronaut=rgb2gray(data.astronaut())
base = np.tile(astronaut, (16, 16))
pyramid = list(pyramid_gaussian(base, downscale=2, multichannel=False))[:-8]
print([p.shape[:2] for p in pyramid])

with app_context():
    # create the viewer
    viewer = napari.Viewer()

    # add image pyramid
    viewer.add_pyramid(pyramid)

    camera = viewer.window.qt_viewer.view.camera
    camera.rect = (-.1*pyramid[0].shape[1], 0, 1.2*pyramid[0].shape[1],
                   pyramid[0].shape[0])
