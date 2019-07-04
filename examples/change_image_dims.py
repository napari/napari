"""
Change image shape and dims directly on the layer
"""

import numpy as np
from skimage import data
from napari import Viewer
from napari.util import app_context


with app_context():
    viewer = Viewer()

    # add data
    blobs = data.binary_blobs(
        length=128, blob_size_fraction=0.05, n_dim=3, volume_fraction=0.25
    ).astype(float)

    layer = viewer.add_image(blobs[:64])

    # switch number of displayed dimensions
    layer.data = blobs[0]

    # switch number of displayed dimensions
    layer.data = blobs[:64]

    # switch the shape of the displayed data
    layer.data = blobs[:3]
