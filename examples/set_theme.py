"""
Displays an image and sets the theme to 'light'.
"""

from skimage import data
import napari


with napari.gui_qt():
    # create the viewer with an image
    viewer = napari.view_image(
        data.astronaut(), rgb=True, name='astronaut'
    )

    # set the theme to 'light'
    viewer.style.theme = 'light'
