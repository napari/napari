"""
Display one 4-D image layer using the add_image API
"""

from skimage import data
from skimage.morphology import binary_dilation
from scipy import ndimage as ndi
import numpy as np
import napari


with napari.gui_qt():
    np.random.seed(1)
    viewer = napari.Viewer()
    blobs = data.binary_blobs(length=128, volume_fraction=0.1, n_dim=2)
    labeled = ndi.label(blobs)[0]
    labels_layer = viewer.add_labels(labeled, name='blob ID')

    @labels_layer.mouse_drag_callbacks.append
    def get_connected_component_shape(layer, event):
        cords = np.round(layer.coordinates).astype(int)
        val = layer.get_value()
        binary = layer.data == val
        binary = binary_dilation(binary)
        size = np.sum(binary)
        data = layer.data
        data[binary] = val
        layer.data = data
        msg = f'clicked at {cords} on blob {val} which is  {size} now pixels large'
        layer.status = msg
        print(msg)
