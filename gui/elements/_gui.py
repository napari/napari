from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout

from ._viewer import Viewer


class Gui:
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

        # self._viewers = []
        self.viewer = Viewer()
        self._qt_central_widget.layout().addWidget(self.viewer._qt)

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
