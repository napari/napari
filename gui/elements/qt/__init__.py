"""
Custom Qt widgets that serve as native objects that the public-facing elements
wrap.
"""
from PyQt5.QtWidgets import QApplication as QtApplication

from ._layerPanel import QtLayerPanel
from ._controls import QtControls
from ._imageLayer import QtImageLayer
from ._markersLayer import QtMarkersLayer
from ._center import QtCenter
from ._viewer import QtViewer

from vispy import app
app.use_app('pyqt5')
del app
