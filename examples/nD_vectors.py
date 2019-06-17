"""
Display one 4-D image layer using the add_image API
"""

import numpy as np
from skimage import data
import napari
from napari.util import app_context


with app_context():
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

    # sample vector coord-like data
    n = 500
    pos = np.zeros((n, 2, 2), dtype=np.float32)
    phi_space = np.linspace(0, 4 * np.pi, n)
    radius_space = np.linspace(0, 20, n)

    # assign x-y position
    pos[:, 0, 0] = radius_space * np.cos(phi_space) + 64
    pos[:, 0, 1] = radius_space * np.sin(phi_space) + 64

    # assign x-y projection
    pos[:, 1, 0] = 2 * radius_space * np.cos(phi_space)
    pos[:, 1, 1] = 2 * radius_space * np.sin(phi_space)

    # add the broadcast vectors
    layer = viewer.add_vectors(pos, width=0.4, name='broadcast vectors')

    planes = np.round(np.linspace(0, 128, 500)).astype(int)
    planes = np.concatenate(
        (planes.reshape((500, 1, 1)), np.zeros((500, 1, 1))), axis=1
    )
    vectors = np.concatenate((planes, pos), axis=2)

    # add the sliced vectors
    layer = viewer.add_vectors(
        vectors, width=0.4, name='sliced vectors', color='blue'
    )
