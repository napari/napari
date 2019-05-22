"""
Change image shape directly on the layer
"""

import numpy as np
from skimage import data
from napari import ViewerApp
from napari.util import app_context


with app_context():
    viewer = ViewerApp()

    # add 3D data
    blobs = data.binary_blobs(length=128, blob_size_fraction=0.05,
                              n_dim=3, volume_fraction=.25).astype(float)

    layer = viewer.add_image(blobs[:64])

    # switch to 3D data
    layer.image = blobs[:3]
    #viewer._on_layers_change(None)
