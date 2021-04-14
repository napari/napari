"""
Display one image using the add_image API.
"""

from skimage import data
import napari


# create the viewer with an image
viewer = napari.view_image(data.astronaut(), rgb=True)

napari.run()
