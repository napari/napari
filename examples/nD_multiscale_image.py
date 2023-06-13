"""
nD multiscale image
===================

Displays an nD multiscale image

.. tags:: visualization-advanced
"""

import numpy as np
from skimage.transform import pyramid_gaussian

import napari

# create multiscale from random data
base = np.random.random((1536, 1536))
base = np.array([base * (8 - i) / 8 for i in range(8)])
print('base shape', base.shape)
multiscale = list(
    pyramid_gaussian(base, downscale=2, max_layer=2, channel_axis=-1)
)
print('multiscale level shapes: ', [p.shape for p in multiscale])

# add image multiscale
viewer = napari.view_image(multiscale, contrast_limits=[0, 1], multiscale=True)

if __name__ == '__main__':
    napari.run()
