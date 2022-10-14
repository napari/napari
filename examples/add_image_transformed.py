"""
Add image transformed
=====================

Display one image and transform it using the :func:`view_image` API.

.. tags:: visualization-basic
"""

from skimage import data
import napari

# create the viewer with an image and transform (rotate) it
viewer = napari.view_image(data.astronaut(), rgb=True, rotate=45)

if __name__ == '__main__':
    napari.run()
