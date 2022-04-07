"""
nD volume
=========

Slide through 3D Volume series in 4D data using the add_volume API

"""

from skimage.data import binary_blobs
import numpy as np
import napari


blobs = np.asarray(
    [
        binary_blobs(length=64, volume_fraction=0.1, n_dim=3).astype(float)
        for i in range(10)
    ]
)
viewer = napari.Viewer(ndisplay=3)

# add the volume
layer = viewer.add_image(blobs)

if __name__ == '__main__':
    napari.run()
