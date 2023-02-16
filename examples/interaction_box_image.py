"""
Interaction box image
=====================

This example demonstrates activating 'transform' mode on the image layer.
This allows the user to manipulate the image via the interaction box
(blue box and points around the image).

.. tags:: experimental
"""

from skimage import data

import napari

viewer = napari.view_image(data.astronaut(), rgb=True)
viewer.layers.selection.active.mode = 'transform'

if __name__ == '__main__':
    napari.run()
