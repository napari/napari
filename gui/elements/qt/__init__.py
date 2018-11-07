"""
Qt backend.
"""
from PyQt5.QtWidgets import QApplication as QtApplication

from ._window import QtWindow
from ._viewer import QtViewer
from ._layerList import QtLayerList
from ._controls import QtControls
from ._imageLayer import QtImageLayer
from ._markerLayer import QtMarkerLayer


from vispy import app
app.use_app('pyqt5')
del app
