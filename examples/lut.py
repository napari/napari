"""
Display a single grayscale image using the imshow API and then adjust the color
limits and the colormap. Collectively these properties can be referred to as
the look-up-table or lut.
"""

import sys
from PyQt5.QtWidgets import QApplication

from skimage import data
from skimage.color import rgb2gray
from napari import Window, Viewer


# starting
application = QApplication(sys.argv)

# show grayscale image
viewer = Viewer()
window = Window(viewer)
# add the image
image = rgb2gray(data.astronaut())
viewer.add_image(image)
layer = viewer.layers[0]
# change the clims
layer.clim = [0.25, .75]
# change the colormap
layer.colormap = 'gray'

sys.exit(application.exec())
