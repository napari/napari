"""
Display a single rgb colored image using the imshow API
"""

import sys
from PyQt5.QtWidgets import QApplication

from skimage import data
from napari_gui import imshow


# starting
application = QApplication(sys.argv)

# show rgb image
image = data.astronaut()
viewer = imshow(image, {})

sys.exit(application.exec())
