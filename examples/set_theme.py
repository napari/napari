"""
Displays an image and sets the theme to 'light'.
"""

from skimage import data
from napari import ViewerApp
from napari.util import app_context


with app_context():
    # create the viewer with an image
    viewer = ViewerApp(astronaut=data.astronaut(),
                       title='napari light theme')

    # set the theme to 'light'
    viewer.theme = 'light'

