"""
Display one 3-D volume layer using the add_volume API
"""

from skimage import data
import napari


blobs = data.binary_blobs(length=64, volume_fraction=0.1, n_dim=3).astype(
    float
)
viewer = napari.Viewer(ndisplay=3)
# add the volume
viewer.add_image(blobs, scale=[3, 1, 1])

napari.run()
