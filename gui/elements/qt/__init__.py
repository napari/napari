"""
Custom Qt widgets that serve as native objects that the public-facing elements
wrap.
"""
from PyQt5.QtWidgets import QApplication as QtApplication

from ._layerPanel import QtLayerPanel
from ._controlBars import QtControlBars
from ._imageLayer import QtImageLayer
from ._markersLayer import QtMarkersLayer
from ._dimensions import QtDimensions
from ._viewer import QtViewer
from ._vectorsLayer import QtVectorsLayer

from vispy import app
app.use_app('pyqt5')
del app
