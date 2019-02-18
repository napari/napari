from vispy.util.event import EmitterGroup, Event

from .view import QtControls


class Controls:
    """Controls object.

    Parameters
    ----------
    viewer : Viewer
        Parent viewer.
    """
    def __init__(self, viewer):
        self.viewer = viewer

        self._qt = QtControls(self)

    def update(self):
        layer = self.viewer._top
        if layer is not None:
            self._qt.display(layer)
            self.viewer.status = layer.status
            self.viewer.help = layer.help
            self.viewer.cursor = layer.cursor
            self.viewer.interactive = layer.interactive
        else:
            self._qt.display(None)
            self.viewer.status = 'Ready'
            self.viewer.help = ''
            self.viewer.cursor = 'standard'
            self.viewer.interactive = True
