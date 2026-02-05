"""
Grid mode
=========

Display layers in a grid using the `viewer.canvas.grid` API. When grid is enabled,
layers are automatically arranged - based on the stride, height, and width parameters -
in a grid of viewboxes linked to the main Camera and Dims. Viewer overlays such as
scale_bar can also be shown in a gridded manner.

.. tags:: visualization-basic
"""

from skimage import data

import napari

viewer = napari.Viewer()
layers = viewer.add_image(data.lily(), channel_axis=2)

viewer.canvas.grid.enabled = True
# a stride of 2 means that two consecutive layers are placed in each
# viewbox instead of just one
viewer.canvas.grid.stride = 2
# we leave width and height as -1, automatically resulting in a square grid

# setting the spacing to a value between 0 and 1 adds a padding between
# viewboxes relative to their sizes. Setting to a value greater than 1
# results in a padding of that exact number of pixels instead.
viewer.canvas.grid.spacing = 0.1

viewer.canvas.scale_bar.visible = True
viewer.canvas.scale_bar.box = True
# show scalebar in each grid instead of just once
viewer.canvas.scale_bar.gridded = True

if __name__ == '__main__':
    napari.run()
