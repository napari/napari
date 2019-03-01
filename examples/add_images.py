"""
Display two image layers using the add_image API
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

sys.exit(application.exec())
