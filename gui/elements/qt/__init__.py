"""
Qt backend.
"""
from PyQt5.QtWidgets import QApplication as QtApplication

from ._window import QtWindow
from ._viewer import QtViewer


from vispy import app
app.use_app('pyqt5')
del app
