"""
Interaction box image
=====================

Demonstrate interaction box on image layer
"""

from skimage import data
import numpy as np
import napari
from napari.utils.transforms import Affine

viewer = napari.view_image(data.astronaut(), rgb=True)
viewer.layers.selection.active.mode = 'transform'

if __name__ == '__main__':
    napari.run()
