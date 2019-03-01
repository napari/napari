"""
Add four image layers using the add_image API and then reorder them using the
layers swap method and remove one using the layers pop method
"""

import sys
from PyQt5.QtWidgets import QApplication

from skimage import data
from skimage.color import rgb2gray
from napari_gui import Window, Viewer


# starting
application = QApplication(sys.argv)

# create the viewer and window
viewer = Viewer()
window = Window(viewer)
# add the first image
viewer.add_image(rgb2gray(data.astronaut()), {})
# add the second image
viewer.add_image(data.camera(), {})
# add the third image
viewer.add_image(data.coins(), {})
# add the fourth image
viewer.add_image(data.moon(), {})

# Swap the astronaut and coins
viewer.layers.swap(0, 2)
# Swap the coins and moon
viewer.layers.swap(0, 3)
# Remove the camera
viewer.layers.remove(viewer.layers[1])

sys.exit(application.exec())
