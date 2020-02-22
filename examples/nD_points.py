"""
Display one points layer on top of one 4-D image layer using the
add_points and add_image APIs, where the markes are visible as nD objects
accross the dimensions, specified by their size
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
    viewer = napari.view_image(blobs.astype(float))

    # add the points
    points = np.array(
        [
            [0, 0, 100, 100],
            [0, 0, 50, 120],
            [1, 0, 100, 40],
            [2, 10, 110, 100],
            [9, 8, 80, 100],
        ]
    )
    viewer.add_points(points, size=[0, 6, 10, 10], face_color='blue', n_dimensional=True)
