"""
Add grayscale image
===================

Display one grayscale image using the add_image API.

.. tags:: visualization-basic
"""

from skimage import data
import napari
import numpy as np


# simulating a grayscale image here for testing contrast limits adjustments
image = data.astronaut().mean(-1) * 100 + 100
image += np.random.rand(*image.shape) * 3000
viewer = napari.view_image(image.astype(np.uint16))

if __name__ == '__main__':
    napari.run()
