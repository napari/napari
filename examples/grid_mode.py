"""
Grid mode
=========

Display layers in a grid using the `viewer.grid` API. When grid is enabled,
layers are arranged automatically arranged in a grid of viewboxes linked to
the main Camera and Dims. Viewer overlays such as scale_bar can also be shown
in a gridded manner.

.. tags:: visualization-basic
"""

from skimage import data

import napari

viewer = napari.Viewer()
layers = viewer.add_image(data.lily(), channel_axis=2)

viewer.grid.enabled = True
# a stride of 2 means that two consecutive layers are placed in each
# viewbox instead of just one
viewer.grid.stride = 2

viewer.scale_bar.visible = True
viewer.scale_bar.box = True
# show scalebar in each grid instead of just once
viewer.scale_bar.gridded = True

if __name__ == '__main__':
    napari.run()
