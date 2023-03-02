"""
nD multiscale image non-uniform
===============================

Displays an nD multiscale image

.. tags:: visualization-advanced
"""

import numpy as np
from skimage import data
from skimage.transform import pyramid_gaussian

import napari

# create multiscale from astronaut image
astronaut = data.astronaut()
base = np.tile(astronaut, (3, 3, 1))
multiscale = list(
    pyramid_gaussian(base, downscale=2, max_layer=3, channel_axis=-1)
)
multiscale = [
    np.array([p * (abs(3 - i) + 1) / 4 for i in range(6)]) for p in multiscale
]
print('multiscale level shapes: ', [p.shape for p in multiscale])

# add image multiscale
viewer = napari.view_image(multiscale, multiscale=True)

if __name__ == '__main__':
    napari.run()
