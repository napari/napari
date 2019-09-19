"""
Display multiple image layers using the add_image API and then reorder them
using the layers swap method and remove one
"""

from skimage import data
from skimage.color import rgb2gray
import napari


with napari.gui_qt():
    # create the viewer with several image layers
    viewer = napari.add_image(rgb2gray(data.astronaut()), name='astronaut')
    viewer.add_image(data.camera(), name='photographer')
    viewer.add_image(data.coins(), name='coins')
    viewer.add_image(data.moon(), name='moon')

    # remove the coins layer
    viewer.layers.remove('coins')

    # swap the order of astronaut and moon
    viewer.layers['astronaut', 'moon'] = viewer.layers['moon', 'astronaut']
