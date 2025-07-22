"""
Grid mode
=========

Display layers in a grid using the `viewer.grid` API.

.. tags:: visualization-basic
"""

from skimage import data

import napari

viewer = napari.Viewer()
layers = viewer.add_image(data.lily(), channel_axis=2)

viewer.grid.enabled = True

# show scalebar in gridded mode
viewer.scale_bar.visible = True
viewer.scale_bar.gridded = True
viewer.scale_bar.box = True

if __name__ == '__main__':
    napari.run()
