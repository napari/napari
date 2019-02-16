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
        #iteration goes backwards to find top most selected layer
        for layer in self.viewer.layers[::-1]:
            if layer.selected:
                self._qt.display(layer)
                self.viewer.status = layer._controls_msg
                break
        else:
            self._qt.display(None)
            self.viewer.status = 'Ready'
