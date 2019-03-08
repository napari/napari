# This is in interactive test

import sys
from time import sleep

import threading
from PyQt5.QtWidgets import QApplication

from napari import Viewer
from napari.components import Dims
from napari.components._dims.model import DimsMode
from napari.components._dims.view import QtDims

# starts the QT event loop
from napari.components._viewer.view import QtViewer

app = QApplication(sys.argv)


viewer  = Viewer()

# creates a widget to view (and control) the model:
widget = QtViewer(viewer)

# makes the view visible on the desktop:
widget.show()

# Start the QT event loop.
sys.exit(app.exec())
