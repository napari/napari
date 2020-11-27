"""MonitorCommands class.
"""

from ....layers.image.experimental.octree_image import OctreeImage
from ...layerlist import LayerList


class MonitorCommands:
    """Commands that remote client can call.

    One Monitor API method is called 'command_queue'. It returns a Queue
    proxy object. Remote clients can insert "commands" into that queue.

    When MonitorApi.poll() is called, it checks the queue. If the queue
    item is a dict, then the top-level keys of that dict are expected to be
    methods in this MonitorCommands class. The value of the key will be the
    argument to the method.

    Parameters
    ----------
    layers : LayerList
        The viewer's layers, so we can modify them.

    Notes
    -----
    This is kind of a crude event or call back system. It proves we can
    receive and execute commands from remote clients. Probably we want
    something integrated with our event system? Something so the code
    processing commands can be distributed throughout the system and is not
    centralized here.
    """

    def __init__(self, layers: LayerList):
        self.layers = layers

    def show_grid(self, show: bool) -> None:
        """Set whether the octree tile grid is visible.

        Parameters
        ----------
        show : bool
            If True the grid is shown.
        """
        for layer in self.layers.selected:
            if isinstance(layer, OctreeImage):
                layer.show_grid = show
