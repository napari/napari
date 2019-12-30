"""
Displays an image pyramid
"""

import numpy as np
from skimage import data
from skimage.transform import pyramid_gaussian

import napari


# create pyramid from astronaut image
base = np.tile(data.astronaut().sum(-1), (8, 8))
pyramid = [
    x.astype(np.complex)
    for x in list(
        pyramid_gaussian(base, downscale=2, max_layer=4, multichannel=False)
    )
]


with napari.gui_qt():
    # add image pyramid
    napari.view_image(pyramid, is_pyramid=True)
