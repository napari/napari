"""
Custom Qt widgets that serve as native objects that the public-facing elements
wrap.
"""
from qtpy.QtWidgets import QApplication as QtApplication
from qtpy import API_NAME
from os import environ
from .view import Window

from vispy import app

# set vispy to use same backend as qtpy
app.use_app(API_NAME)

del app
