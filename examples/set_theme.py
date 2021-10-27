"""
Displays an image and sets the theme to 'light'.
"""

from skimage import data
import napari


# create the viewer with an image
viewer = napari.view_image(data.astronaut(), rgb=True, name='astronaut')

# set the theme to 'light'
viewer.theme = 'light'

napari.run()