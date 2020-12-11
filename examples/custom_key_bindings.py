"""
Display one 4-D image layer using the add_image API
"""

from skimage import data
import napari


with napari.gui_qt():
    blobs = data.binary_blobs(
        length=128, blob_size_fraction=0.05, n_dim=2, volume_fraction=0.25
    ).astype(float)

    viewer = napari.view_image(blobs, name='blobs')

    @viewer.bind_key('a')
    def accept_image(viewer):
        msg = 'this is a good image'
        viewer.info.status = msg
        print(msg)
        next(viewer)

    @viewer.bind_key('r')
    def reject_image(viewer):
        msg = 'this is a bad image'
        viewer.info.status = msg
        print(msg)
        next(viewer)

    def next(viewer):
        blobs = data.binary_blobs(
            length=128, blob_size_fraction=0.05, n_dim=2, volume_fraction=0.25
        ).astype(float)
        viewer.layers[0].data = blobs

    @napari.Viewer.bind_key('w')
    def hello(viewer):
        # on press
        viewer.info.status = 'hello world!'

        yield

        # on release
        viewer.info.status = 'goodbye world :('

    # change viewer title
    viewer.info.title = 'quality control images'
