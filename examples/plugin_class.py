"""
Display two image layers using the add_image API
"""

import sys
from PyQt5.QtWidgets import QApplication

from skimage import data
from napari import Window, Viewer
from napari.plugins import GaussianBlur

# starting
application = QApplication(sys.argv)

# create the viewer and window
viewer = Viewer()
window = Window(viewer)
# add the first image
viewer.add_image(data.camera())
# add the second image
viewer.add_image(data.camera())
# add the plugin
viewer.add_plugin(GaussianBlur())

sys.exit(application.exec())
