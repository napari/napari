"""
Display one markers layer ontop of one 4-D image layer using the
add_markers and add_image APIs, where the markes are visible as nD objects
accross the dimensions, specified by their size
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
# add the markers
markers = np.array([[200, 200, 0, 0], [50, 150, 0, 0], [100, 400, 1, 0],
                   [300, 200, 0, 1], [400, 100, 0, 1]])
viewer.add_markers(markers, size=[10, 10, 6, 0], face_color='blue',
                   n_dimensional=True)

sys.exit(application.exec())
