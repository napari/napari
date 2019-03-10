"""
Display multiple image layers using the add_image API and then reorder them
using the layers swap method and remove one
"""

from skimage import data
from skimage.color import rgb2gray
from napari import ViewerApp
from napari.util import app_context


with app_context():
    # create the viewer and window
    viewer = ViewerApp()

    # add the images
    viewer.add_image(rgb2gray(data.astronaut()), name='astronaut')
    viewer.add_image(data.camera(), name='photographer')
    viewer.add_image(data.coins(), name='coins')
    viewer.add_image(data.moon(), name='moon')

    # remove the coins layer
    viewer.remove_layer('coins')

    # swap the order of astronaut and moon
    viewer.layers['astronaut', 'moon'] = viewer.layers['moon', 'astronaut']
