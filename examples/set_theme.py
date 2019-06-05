"""
Displays an image and sets the theme to 'light'.
"""

from skimage import data
import napari
from napari.util import app_context


with app_context():
    # create the viewer with an image
    viewer = napari.view(astronaut=data.astronaut(), multichannel=True, title='napari')

    # set the theme to 'light'
    viewer.theme = 'light'
