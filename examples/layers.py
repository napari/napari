"""
Display multiple image layers using the add_image API and then reorder them
using the layers swap method and remove one
"""

from skimage import data
from skimage.color import rgb2gray
from napari import view, gui_qt


with gui_qt():
    # create the viewer with several image layers
    viewer = view(
        astronaut=rgb2gray(data.astronaut()),
        photographer=data.camera(),
        coins=data.coins(),
        moon=data.moon(),
    )

    # remove the coins layer
    viewer.layers.remove('coins')

    # swap the order of astronaut and moon
    viewer.layers['astronaut', 'moon'] = viewer.layers['moon', 'astronaut']
