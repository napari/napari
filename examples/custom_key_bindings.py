"""
Custom key bindings
===================

Display one 4-D image layer using the ``add_image`` API

.. tags:: gui
"""

from skimage import data

import napari

blobs = data.binary_blobs(
    length=128, blob_size_fraction=0.05, n_dim=2, volume_fraction=0.25
).astype(float)

viewer = napari.view_image(blobs, name='blobs')


@viewer.bind_key('a')
def accept_image(viewer):
    msg = 'this is a good image'
    viewer.status = msg
    print(msg)
    set_layer_data(viewer)


@viewer.bind_key('r')
def reject_image(viewer):
    msg = 'this is a bad image'
    viewer.status = msg
    print(msg)
    set_layer_data(viewer)


def set_layer_data(viewer):
    blobs = data.binary_blobs(
        length=128, blob_size_fraction=0.05, n_dim=2, volume_fraction=0.25
    ).astype(float)
    viewer.layers[0].data = blobs


@napari.Viewer.bind_key('w')
def hello(viewer):
    # on press
    viewer.status = 'hello world!'

    yield

    # on release
    viewer.status = 'goodbye world :('


# change viewer title
viewer.title = 'quality control images'

if __name__ == '__main__':
    napari.run()
