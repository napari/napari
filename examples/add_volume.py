"""
Display one 3-D volume layer using the add_volume API
"""

from skimage import data
import napari


with napari.gui_qt():
    blobs = data.binary_blobs(length=64, volume_fraction=0.1, n_dim=3).astype(
        float
    )
    viewer = napari.Viewer()
    # add the volume
    viewer.add_volume(blobs, spacing=[3, 1, 1])
