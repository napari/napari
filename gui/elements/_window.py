from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout

from ._viewer import Viewer


class Window:
    """Application window.
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
        """Adds a viewer to the containing layout.

        Returns
        -------
        viewer : Viewer
            Viewer object.
        """
        viewer = Viewer(self)
        self.viewers.append(viewer)
        self._qt_central_widget.layout().addLayout(viewer.controls._qt)
        self._qt_central_widget.layout().addWidget(viewer._qt)
        self._qt_central_widget.layout().addWidget(viewer.layers._qt)
        return viewer

    def resize(self, *args):
        self._qt_window.resize(*args)

    def show(self):
        self.resize(self._qt_window.layout().sizeHint())
        self._qt_window.show()
        self._qt_window.raise_()
