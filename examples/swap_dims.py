"""
Swap dims
=========

Display a 4-D image and points layer and swap the displayed dimensions

.. tags:: visualization-nD
"""

import numpy as np
from skimage import data

import napari

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
        [0, 0, 0, 100],
        [0, 0, 50, 120],
        [1, 0, 100, 40],
        [2, 10, 110, 100],
        [9, 8, 80, 100],
    ]
)
viewer.add_points(
    points, size=[0, 6, 10, 10], face_color='blue', out_of_slice_display=True
)

viewer.dims.order = (0, 2, 1, 3)

if __name__ == '__main__':
    napari.run()
