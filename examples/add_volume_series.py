"""
Display one 3-D volume layer using the add_volume API
"""

from skimage import data
import napari


with napari.gui_qt():
    blobs = data.binary_blobs(length=128, blob_size_fraction=0.05, n_dim=4)
    blobs = blobs.astype(float) * 255
    viewer = napari.Viewer()
    for i in range(blobs.shape[0]):
        # add the volume
        layer = viewer.add_volume(blobs[i])

        if i == 0:
            layer.visible = True
