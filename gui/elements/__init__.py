"""PyQt5 elements.
"""
from .image_window import ImageWindow
from .image_widget import ImageViewerWidget
from .image_container import ImageContainer


from vispy import app
app.use_app('pyqt5')
del app
