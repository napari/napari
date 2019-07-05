"""
Display one 4-D image layer using the add_image API
"""

import numpy as np
from skimage import data
import napari
from napari.util import gui_qt


with gui_qt():
    blobs = np.stack(
        [
            data.binary_blobs(
                length=128, blob_size_fraction=0.05, n_dim=3, volume_fraction=f
            )
            for f in np.linspace(0.05, 0.5, 10)
        ],
        axis=0,
    )
    viewer = napari.view(blobs.astype(float))
