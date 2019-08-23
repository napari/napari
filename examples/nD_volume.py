"""
Slide through 3D Volume series in 4D data using the add_volume API
"""

from skimage import data
import numpy as np
import napari


with napari.gui_qt():
    blobs = np.asarray(
        [
            data.binary_blobs(length=64, volume_fraction=0.1, n_dim=3).astype(
                float
            )
            for i in range(10)
        ]
    )
    viewer = napari.Viewer()
    # add the volume
    layer = viewer.add_image(blobs, ndisplay=3)
