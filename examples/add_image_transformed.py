"""
Add image transformed
=====================

Display one image and transform it using the :meth:`add_image` API.
"""

from skimage import data
import napari


# create a viewer
viewer = napari.Viewer()
# add and transform image
viewer.add_image(data.astronaut(), rgb=True, rotate=45)

if __name__ == '__main__':
    napari.run()
