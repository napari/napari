"""
Paint nD
========

Display a 4D labels layer and paint only in 3D.

This is useful e.g. when proofreading segmentations within a time series.

.. tags:: analysis
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
viewer = napari.view_image(blobs.astype(float), rendering='attenuated_mip')
labels = viewer.add_labels(np.zeros_like(blobs, dtype=np.int32))
labels.n_edit_dimensions = 3
labels.brush_size = 15
labels.mode = 'paint'
labels.n_dimensional = True

if __name__ == '__main__':
    napari.run()
