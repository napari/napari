"""
Tiled rendering 2D
==================

This example shows how to display tiled, chunked data in napari using the
experimental octree support.

If given a large 2D image with octree support enabled, napari will only load
and display the tiles in the center of the current canvas view. (Note: napari
uses its own internal tile size that may or may not be aligned with the
underlying tiled data, but this should have only minor performance
consequences.)

If octree support is *not* enabled, napari will try to load the entire image,
which may not fit in memory and may bring your computer to a halt. Oops! So, we
make sure that we enable octree support by setting the NAPARI_OCTREE
environment variable to 1 if it is not set by the user.

.. tags:: experimental
"""

import os

# important: if this is not set, the entire ~4GB array will be created!
os.environ.setdefault('NAPARI_OCTREE', '1')

import dask.array as da # noqa: E402
import napari   # noqa: E402


ndim = 2
data = da.random.randint(
        0, 256, (65536,) * ndim,
        chunks=(256,) * ndim,
        dtype='uint8'
        )

viewer = napari.Viewer()
viewer.add_image(data, contrast_limits=[0, 255])
# To turn off grid lines
#viewer.layers[0].display.show_grid = False

# set small zoom so we don't try to load the whole image at once
viewer.camera.zoom = 0.75

# run the example — try to pan around!
if __name__ == '__main__':
    napari.run()
