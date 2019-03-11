"""
Display one 4-D image layer using the add_image API
"""

import numpy as np
from skimage import data
from napari import ViewerApp
from napari.util import app_context


with app_context():
    # TODO: change axis back to 0 once dims displayed in sane order
    blobs = np.stack([data.binary_blobs(length=128, blob_size_fraction=0.05,
                                        n_dim=3, volume_fraction=f)
                     for f in np.linspace(0.05, 0.5, 10)], axis=-1)
    viewer = ViewerApp(blobs.astype(float))
