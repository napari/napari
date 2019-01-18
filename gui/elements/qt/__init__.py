"""
Custom Qt widgets that serve as native objects that the public-facing elements
wrap.
"""
from PyQt5.QtWidgets import QApplication as QtApplication

from ._layer_panel import QtLayerPanel
from ._control_bars import QtControlBars
from ._dimensions import QtDimensions
from ._viewer import QtViewer
from ._vectorsLayer import QtVectorsLayer

from vispy import app
app.use_app('pyqt5')
del app
