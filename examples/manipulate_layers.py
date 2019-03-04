"""
Display multiple image layers using the add_image API and then reorder them
using the layers swap method and remove one
"""

import sys
from PyQt5.QtWidgets import QApplication

from skimage import data
from skimage.color import rgb2gray
from napari import Window, Viewer


# starting
application = QApplication(sys.argv)

# create the viewer and window
viewer = Viewer()
window = Window(viewer)
# add the first image
viewer.add_image(rgb2gray(data.astronaut()))
viewer.layers[-1].name = 'astronaut'
# add the second image
viewer.add_image(data.camera())
viewer.layers[-1].name = 'photographer'
# add the third image
viewer.add_image(data.coins())
viewer.layers[-1].name = 'coins'
# add the fourth image
viewer.add_image(data.moon())
viewer.layers[-1].name = 'moon'

# remove the coins
viewer.layers.remove('coins')

# swap the order of camera and moon
viewer.layers['photographer', 'moon'] = viewer.layers['moon', 'photographer']

# turn off the visibility of the photographer
#viewer.layers['photographer'].visible = False

sys.exit(application.exec())
