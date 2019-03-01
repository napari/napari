"""
Display one 4-D image layer using the add_image API
"""

import sys
from PyQt5.QtWidgets import QApplication

import numpy as np
from skimage import data
from skimage.color import rgb2gray
from napari_gui import Window, Viewer


# starting
application = QApplication(sys.argv)

# create the viewer and window
viewer = Viewer()
window = Window(viewer)
# add the image
viewer.add_image(np.random.rand(500, 500, 20, 10), {})

sys.exit(application.exec())
