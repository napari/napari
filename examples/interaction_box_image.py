"""
Interaction box image
=====================

This example demonstrates activating 'transform' mode on the image layer.
This allows the user to manipulate the image via the interaction box
(blue box and points around the image).
"""

from skimage import data
import numpy as np
import napari
from napari.utils.transforms import Affine

viewer = napari.view_image(data.astronaut(), rgb=True)
viewer.layers.selection.active.mode = 'transform'

if __name__ == '__main__':
    napari.run()
