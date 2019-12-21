"""
Displays an image pyramid
"""

from skimage.transform import pyramid_gaussian
import napari
import numpy as np


# create pyramid from random data
base = np.random.random((1536, 1536))
base = np.array([base * (8 - i) / 8 for i in range(8)])
print('base shape', base.shape)
pyramid = list(
    pyramid_gaussian(base, downscale=2, max_layer=2, multichannel=False)
)
print('pyramid level shapes: ', [p.shape for p in pyramid])

with napari.gui_qt():
    # add image pyramid
    napari.view_image(pyramid, contrast_limits=[0, 1], is_pyramid=True)
