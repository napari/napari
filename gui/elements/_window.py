from .qt import QtWindow

from ._viewer import Viewer


class Window:
    """Application window.
    """
    def __init__(self):
        self._qt = QtWindow()
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
        self._qt.add_viewer(viewer)
        return viewer

    def resize(self, width, height):
        self._qt.resize(width, height)

    def show(self):
        self.resize(self._qt.layout().sizeHint())
        self._qt.show()
        self._qt.raise_()
