"""
Display a single image layer using the add_image API
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
# add the image
viewer.add_image(rgb2gray(data.astronaut()))

sys.exit(application.exec())
