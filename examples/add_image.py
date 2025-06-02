"""
Add image
=========

Display one image using the :func:`add_image` API.

.. tags:: visualization-basic
"""

from skimage import data

import napari

# create the viewer
viewer = napari.Viewer()

# add the image to the viewer instance
viewer.add_image(data.astronaut(), rgb=True)

if __name__ == '__main__':
    napari.run()
