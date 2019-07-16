"""
Display one 3-D volume layer using the add_volume API
"""

from skimage import data
import napari


with napari.gui_qt():
    blobs = data.binary_blobs(length=128, blob_size_fraction=0.05, n_dim=3)
    blobs = blobs.astype(float) * 255
    viewer = napari.Viewer()
    # add the volume
    viewer.add_volume(blobs)
