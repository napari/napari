"""
Display a labels layer with classes for each label, including the background
"""

import napari
import numpy as np
from skimage import data
from scipy import ndimage as ndi
from napari.layers import Labels

blobs = data.binary_blobs(length=128, volume_fraction=0.1, n_dim=2)
labelled, num_labels = ndi.label(blobs)
class_dict = {
    'class': ["Background"]
    + ["Class " + str(i + 1) for i in range(num_labels)]
}

with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_labels(labelled, properties=class_dict)
