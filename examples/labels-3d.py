"""
Display a labels layer above of an image layer using the add_labels and
add_image APIs
"""

from skimage import data
from scipy import ndimage as ndi
from napari import ViewerApp
from napari.util import app_context


with app_context():
    blobs = data.binary_blobs(length=128, volume_fraction=0.1, n_dim=3)
    v = ViewerApp(blobs=blobs)
    v.layers[0].colormap = 'gray'
    labeled = ndi.label(blobs)[0]
    label_layer = v.add_labels(labeled, name='blob ID')
