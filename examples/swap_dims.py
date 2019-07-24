"""
Display one 4-D image layer using the add_image API
"""

import numpy as np
from skimage import data
import napari


with napari.gui_qt():
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

    # add the points
    points = np.array(
        [
            [0, 0, 0, 100],
            [0, 0, 50, 120],
            [1, 0, 100, 40],
            [2, 10, 110, 100],
            [9, 8, 80, 100],
        ]
    )
    viewer.add_points(
        points, size=[0, 6, 10, 10], face_color='blue', n_dimensional=True
    )

    viewer.dims.swap_display(1, 2)
