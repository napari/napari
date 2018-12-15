from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout

from ._viewer import Viewer


class Window:
    """Application window that contains the menu bar and viewers.

    Parameters
    ----------
    viewer : Viewer
        Contained viewer.

    Attributes
    ----------
    viewer : Viewer
        Contained viewer.
    """
    def __init__(self, viewer, show=True):
        self._qt_window = QMainWindow()
        self._qt_center = QWidget()
        self._qt_window.setCentralWidget(self._qt_center)
        self._qt_center.setLayout(QHBoxLayout())
        self._statusBar = self._qt_window.statusBar()
        self._statusBar.showMessage('Ready')

        self.viewer = viewer
        self._qt_center.layout().addWidget(self.viewer._qt)

        if show:
            self.show()

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
