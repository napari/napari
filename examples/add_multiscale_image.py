"""
Displays a multiscale image
"""

from skimage import data
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.transform import pyramid_gaussian
import napari
import numpy as np


# create multiscale from astronaut image
base = np.tile(data.astronaut(), (8, 8, 1))
multiscale = list(
    pyramid_gaussian(base, downscale=2, max_layer=4, multichannel=True)
)
print('multiscale level shapes: ', [p.shape[:2] for p in multiscale])

with napari.gui_qt():
    # add image multiscale
    napari.view_image(multiscale, multiscale=True)
