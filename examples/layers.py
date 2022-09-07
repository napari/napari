"""
Layers
======

Display multiple image layers using the ``add_image`` API and then reorder them
using the layers swap method and remove one

.. tags:: visualization-basic
"""

from skimage import data
from skimage.color import rgb2gray
import numpy as np
import napari


# create the viewer with several image layers
viewer = napari.view_image(rgb2gray(data.astronaut()), name='astronaut')
viewer.add_image(data.camera(), name='photographer')
viewer.add_image(data.coins(), name='coins')
viewer.add_image(data.moon(), name='moon')
viewer.add_image(np.random.random((512, 512)), name='random')
viewer.add_image(data.binary_blobs(length=512, volume_fraction=0.2, n_dim=2), name='blobs')
viewer.grid.enabled = True

if __name__ == '__main__':
    napari.run()
