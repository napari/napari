"""
Custom Qt widgets that serve as native objects that the public-facing elements
wrap.
"""
from qtpy.QtWidgets import QApplication as QtApplication
from os import environ
from .view import Window

from vispy import app

if 'QT_API' in environ:
    app.use_app(environ['QT_API'])
else:
    app.use_app('pyqt5')
del app
