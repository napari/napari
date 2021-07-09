"""
Display one 3-D volume layer using the add_volume API and display it as a plane
"""

from skimage import data
import napari


blobs = data.binary_blobs(length=64, volume_fraction=0.1, n_dim=3).astype(
    float
)
viewer = napari.Viewer(ndisplay=3)
# add the volume
layer = viewer.add_image(blobs)

napari.run()
