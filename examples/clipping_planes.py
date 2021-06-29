"""
Display a 3D image and use a magic gui widget to control clipping planes
"""

from skimage import data
import napari
import numpy as np


blobs = data.binary_blobs(length=64, volume_fraction=0.1, n_dim=3).astype(
    float
)
viewer = napari.Viewer(ndisplay=3)
# add the volume
layer = viewer.add_image(blobs)


layer.clipping_planes = np.tile(0.5, (1, 2, 3))

napari.run()
