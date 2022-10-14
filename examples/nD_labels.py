"""
nD labels
=========

Display a labels layer above of an image layer using the ``add_labels`` and
``add_image`` APIs

.. tags:: visualization-nD
"""

from skimage import data
from scipy import ndimage as ndi
import napari


blobs = data.binary_blobs(length=128, volume_fraction=0.1, n_dim=3)
viewer = napari.view_image(blobs[::2].astype(float), name='blobs', scale=(2, 1, 1))
labeled = ndi.label(blobs)[0]
viewer.add_labels(labeled[::2], name='blob ID', scale=(2, 1, 1))

if __name__ == '__main__':
    napari.run()
