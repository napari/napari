"""
This example shows two image layers using the imshow and add_image
APIs
"""

import sys
from PyQt5.QtWidgets import QApplication

from skimage import data
from skimage.color import rgb2gray
from napari_gui import imshow


if __name__ == '__main__':
    # starting
    application = QApplication(sys.argv)

    # show the first image
    image = rgb2gray(data.astronaut())
    viewer = imshow(image, {})
    # show the second image
    viewer.add_image(data.camera(), {})

    sys.exit(application.exec_())
