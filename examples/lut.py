"""
This example shows a single grayscale image using the imshow API and then
manipulates the color limits and the colormap. Collectively these properties
can be referred to as the look-up-table or lut.
"""

import sys
from PyQt5.QtWidgets import QApplication

from skimage import data
from skimage.color import rgb2gray
from napari_gui import imshow


if __name__ == '__main__':
    # starting
    application = QApplication(sys.argv)

    # show grayscale image
    image = rgb2gray(data.astronaut())
    viewer = imshow(image, {})
    layer = viewer.layers[0]
    # change the clims
    layer.clim = [0.25, .75]
    # change the colormap
    layer.colormap = 'gray'

    sys.exit(application.exec_())
