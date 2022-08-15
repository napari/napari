"""
Add image
=========

Display one image using the :meth:`add_image` API.
"""

from skimage import data
import napari


# create a viewer
viewer = napari.Viewer()
# add image
viewer.add_image(data.astronaut(), rgb=True)

if __name__ == '__main__':
    napari.run()
