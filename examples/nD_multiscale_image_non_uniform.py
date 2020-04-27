"""
Displays an nD multiscale image
"""

from skimage import data
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.transform import pyramid_gaussian
import napari
import numpy as np


# create multiscale from astronaut image
astronaut = data.astronaut()
base = np.tile(astronaut, (3, 3, 1))
multiscale = list(
    pyramid_gaussian(base, downscale=2, max_layer=3, multichannel=True)
)
multiscale = [
    np.array([p * (abs(3 - i) + 1) / 4 for i in range(6)]) for p in multiscale
]
print('multiscale level shapes: ', [p.shape for p in multiscale])

with napari.gui_qt():
    # add image multiscale
    napari.view_image(multiscale, multiscale=True)
