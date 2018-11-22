from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QSplitter

from ._viewer import Viewer


class Window:
    """Application window that contains the menu bar and viewers.

    Attributes
    ----------
    viewers : list of Viewer
        Contained viewers.
    """
    def __init__(self):
        self._qt_window = QMainWindow()
        self._qt_central_widget = QWidget()
        self._qt_window.setCentralWidget(self._qt_central_widget)
        self._qt_central_widget.setLayout(QHBoxLayout())
        self._qt_window.statusBar().showMessage('Ready')

        self._viewers = []

    @property
    def viewers(self):
        """list of Viewer: Contained viewers.
        """
        return self._viewers

    def add_viewer(self):
        """Add a viewer to the containing layout.

        Returns
        -------
        viewer : Viewer
            Viewer object.
        """
        viewer = Viewer(self)
        self.viewers.append(viewer)

        # To split vertical sliders, viewer and layerlist, minimumsizes given for demo purposes/NOT FINAL
        horizontalSplitter = QSplitter(Qt.Horizontal)
        viewer.controls._qt.setMinimumSize(QSize(60, 60))
        horizontalSplitter.addWidget(viewer.controls._qt)
        viewer._qt.setMinimumSize(QSize(100, 100))
        horizontalSplitter.addWidget(viewer._qt)
        viewer.layers._qt.setMinimumSize(QSize(250, 250))
        horizontalSplitter.addWidget(viewer.layers._qt)

        self._qt_central_widget.layout().addWidget(horizontalSplitter)
        return viewer

    def resize(self, width, height):
        """Resize the window.

        Parameters
        ----------
        width : int
            Width in logical pixels.
        height : int
            Height in logical pixels.
        """
        self._qt_window.resize(width, height)

    def show(self):
        """Resize, show, and bring forward the window.
        """
        self._qt_window.resize(self._qt_window.layout().sizeHint())
        self._qt_window.show()
        self._qt_window.raise_()
