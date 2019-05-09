"""
Display one 4-D image layer using the add_image API
"""

from skimage import data
import napari
from napari.util import app_context


with app_context():
    viewer = napari.Viewer()
    blobs = data.binary_blobs(length=128, blob_size_fraction=0.05,
                              n_dim=2, volume_fraction=.25).astype(float)

    viewer.add_image(blobs, name='blobs')

    @viewer.bind_key('a')
    def accept_image(viewer):
        msg = 'this is a good image'
        viewer.status = msg
        print(msg)
        next(viewer)

    @viewer.bind_key('r')
    def reject_image(viewer):
        msg = 'this is a bad image'
        viewer.status = msg
        print(msg)
        next(viewer)

    def next(viewer):
        blobs = data.binary_blobs(length=128, blob_size_fraction=0.05,
                                  n_dim=2, volume_fraction=.25).astype(float)
        viewer.layers[0].image = blobs

    # change viewer title
    viewer.title = 'quality control images'
