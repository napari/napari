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
astronaut = data.astronaut()
base = np.tile(astronaut, (3, 3, 1))
pyramid = list(
    pyramid_gaussian(base, downscale=2, max_layer=3, multichannel=True)
)
pyramid = [
    np.array([p * (abs(3 - i) + 1) / 4 for i in range(6)]) for p in pyramid
]
print('pyramid level shapes: ', [p.shape for p in pyramid])

with napari.gui_qt():
    # create the viewer
    viewer = napari.Viewer()

    # add image pyramid
    viewer.add_pyramid(pyramid, clim_range=[0, 255])
