"""
Add image
=========

Display one image using the :func:`view_image` API.
"""

from skimage import data
import napari


# create the viewer with an image
viewer = napari.view_image(data.astronaut(), rgb=True)

if __name__ == '__main__':
    napari.run()
