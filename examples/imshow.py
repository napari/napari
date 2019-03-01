"""
This example shows a single grayscale image using the imshow API
"""

import sys
from PyQt5.QtWidgets import QApplication

from skimage import data
from skimage.color import rgb2gray
from napari_gui import imshow


# starting
application = QApplication(sys.argv)

# show grayscale image
image = rgb2gray(data.astronaut())
viewer = imshow(image, {})

sys.exit(application.exec())
