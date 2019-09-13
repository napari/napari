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
base = np.tile(astronaut, (3, 3, 1)).astype('float')
base = np.round(np.array([base * (16 - i) / 16 for i in range(16)])).astype(
    np.uint8
)
print('base shape', base.shape)
pyramid = list(
    pyramid_gaussian(base, downscale=2, max_layer=3, rgb=True)
)
print('pyramid level shapes: ', [p.shape[:-1] for p in pyramid])

with napari.gui_qt():
    # create the viewer
    viewer = napari.Viewer()

    # add image pyramid
    viewer.add_pyramid(pyramid, contrast_limits=[0, 255])
