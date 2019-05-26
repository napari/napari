"""
Displays an image pyramid
"""

from skimage import data
from skimage.transform import pyramid_gaussian
import napari
from napari.util import app_context
import numpy as np


image = data.astronaut()
rows, cols, dim = image.shape

# create pyramid from astronaut image
astronaut = data.astronaut().mean(axis=2) / 255
base = np.tile(astronaut, (16, 16))
pyramid = list(pyramid_gaussian(base, downscale=2, max_layer=5,
                                multichannel=False))
pyramid = [(255*p).astype('uint8') for p in pyramid]
print([p.shape[:2] for p in pyramid])

with app_context():
    # create the viewer
    viewer = napari.Viewer()

    # add image pyramid
    viewer.add_pyramid(pyramid, clim_range=[0, 255])

    camera = viewer.window.qt_viewer.view.camera

    # Set the view box of the camera to include the whole base image of the
    # pyramid with a little padding. The view box is a 4-tuple of the x, y
    # corner position followed by width and height
    base_shape = pyramid[0].shape
    camera.rect = (-0.1 * base_shape[1], -0.1 * base_shape[0],
                   1.2 * base_shape[1], 1.2 * base_shape[0])
