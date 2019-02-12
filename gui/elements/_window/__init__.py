"""
Custom Qt widgets that serve as native objects that the public-facing elements
wrap.
"""
from PyQt5.QtWidgets import QApplication as QtApplication

from .view import Window

from vispy import app
app.use_app('pyqt5')
del app
