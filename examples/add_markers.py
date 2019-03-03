"""
Display one markers layer ontop of one image layer using the add_markers and
add_image APIs
"""

import sys
from PyQt5.QtWidgets import QApplication

import numpy as np
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
# add the markers
markers = np.array([[100, 100], [200, 200], [333, 111]])
size = np.array([10, 20, 20])
viewer.add_markers(markers, size=size)

sys.exit(application.exec())
