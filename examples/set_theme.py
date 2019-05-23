"""
Displays an image and sets the theme to 'light'.
"""

from skimage import data
from napari import Viewer
from napari.util import app_context


with app_context():
    # create the viewer with an image
    viewer = Viewer(astronaut=data.astronaut(),
                    title='napari')

    # set the theme to 'light'
    viewer.theme = 'light'
