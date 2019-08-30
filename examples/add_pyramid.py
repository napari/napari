"""
Displays an image pyramid
"""

from skimage import data
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.transform import pyramid_gaussian
import napari
import numpy as np


# create pyramid from astronaut image
base = np.tile(data.astronaut(), (8, 8, 1))
pyramid = list(
    pyramid_gaussian(base, downscale=2, max_layer=4, multichannel=True)
)
print('pyramid level shapes: ', [p.shape[:2] for p in pyramid])

with napari.gui_qt():
    # create the viewer
    viewer = napari.Viewer()

    # add image pyramid
    viewer.add_pyramid(pyramid, contrast_limits_range=[0, 255])
