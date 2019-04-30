"""
Display one 4-D image layer using the add_image API
"""

import numpy as np
from skimage import data
from napari import ViewerApp
from napari.util import app_context


with app_context():
    viewer = ViewerApp()
    blobs = data.binary_blobs(length=128, blob_size_fraction=0.05,
                              n_dim=2, volume_fraction=.25).astype(float)

    viewer.add_image(blobs)

    def accept_image(viewer):
        print('this is a good image')
        next(viewer)

    def reject_image(viewer):
        print('this is a bad image')
        next(viewer)

    def next(viewer):
        blobs = data.binary_blobs(length=128, blob_size_fraction=0.05,
                                  n_dim=2, volume_fraction=.25).astype(float)
        viewer.layers[0].image = blobs

    custom_key_bindings = {'a': accept_image, 'r': reject_image}
    viewer.key_bindings = custom_key_bindings
