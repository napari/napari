"""
Add 3D image
============

Display a 3D image layer using the :meth:`add_image` API.
"""

from skimage import data
import napari


blobs = data.binary_blobs(length=64, volume_fraction=0.1, n_dim=3).astype(
    float
)
viewer = napari.Viewer(ndisplay=3)
# add the volume
viewer.add_image(blobs, scale=[3, 1, 1])

if __name__ == '__main__':
    napari.run()
