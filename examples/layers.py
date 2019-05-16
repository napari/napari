"""
Display multiple image layers using the add_image API and then reorder them
using the layers swap method and remove one
"""

from imageio import imread
from napari import ViewerApp
from napari.util import app_context


astro = imread('imageio:astronaut.png').mean(axis=2) / 255
photographer = imread('imageio:camera.png')
coins = imread('imageio:coins.png')
moon = imread('imageio:moon.png')

with app_context():
    # create the viewer with several image layers
    viewer = ViewerApp(astronaut=astro,
                       photographer=photographer,
                       coins=coins,
                       moon=moon)

    # remove the coins layer
    viewer.layers.remove('coins')

    # swap the order of astronaut and moon
    viewer.layers['astronaut', 'moon'] = viewer.layers['moon', 'astronaut']
