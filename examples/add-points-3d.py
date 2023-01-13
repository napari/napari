"""
Add points 3D
=============

Display a labels layer above of an image layer using the add_labels and
add_image APIs, then add points in 3D

.. tags:: visualization-nD
"""

from skimage import data
from scipy import ndimage as ndi
import napari


blobs = data.binary_blobs(
        length=128, volume_fraction=0.1, n_dim=3
        )[::2].astype(float)
labeled = ndi.label(blobs)[0]

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(blobs, name='blobs', scale=(2, 1, 1))
viewer.add_labels(labeled, name='blob ID', scale=(2, 1, 1))
pts = viewer.add_points()

viewer.camera.angles = (0, -65, 85)
pts.mode = 'add'

if __name__ == '__main__':
    napari.run()
