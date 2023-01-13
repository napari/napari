"""
Add multiscale image
====================

Displays a multiscale image

.. tags:: visualization-advanced
"""

from skimage import data
from skimage.transform import pyramid_gaussian
import napari
import numpy as np


# create multiscale from astronaut image
base = np.tile(data.astronaut(), (8, 8, 1))
multiscale = list(
    pyramid_gaussian(base, downscale=2, max_layer=4, multichannel=True)
)
print('multiscale level shapes: ', [p.shape[:2] for p in multiscale])

# add image multiscale
viewer = napari.view_image(multiscale, multiscale=True)

if __name__ == '__main__':
    napari.run()
