"""
Displays an image and sets the theme to 'light'.
"""

from imageio import imread
from napari import ViewerApp
from napari.util import app_context


astro = imread('imageio:astronaut.png')

with app_context():
    # create the viewer with an image
    viewer = ViewerApp(astronaut=astro,
                       title='napari')

    # set the theme to 'light'
    viewer.theme = 'light'
