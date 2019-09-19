"""
Display a labels layer above of an image layer using the add_labels and
add_image APIs
"""

from skimage import data
from scipy import ndimage as ndi
import napari


with napari.gui_qt():
    blobs = data.binary_blobs(length=128, volume_fraction=0.1, n_dim=3)
    viewer = napari.add_image(blobs.astype(float), name='blobs')
    labeled = ndi.label(blobs)[0]
    viewer.add_labels(labeled, name='blob ID')
