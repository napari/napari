"""
Display multiple image layers using the add_image API and then reorder them
using the layers swap method and remove one
"""

from skimage import data
from skimage.color import rgb2gray
from napari import Window, Viewer
from napari.util import app_context


with app_context():
    # create the viewer and window
    viewer = Viewer()
    window = Window(viewer)

    # add the images
    viewer.add_image(rgb2gray(data.astronaut()), name='astronaut')
    viewer.add_image(data.camera(), name='photographer')
    viewer.add_image(data.coins(), name='coins')
    viewer.add_image(data.moon(), name='moon')

    layers = viewer.layers

    # remove the coins
    layers.remove('coins')

    # swap the order of photographer and moon
    layers['photographer', 'moon'] = layers['moon', 'photographer']
