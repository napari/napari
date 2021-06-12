"""RemoteCommands class.
"""
import json
import logging

from ....layers.image.experimental.octree_image import _OctreeImageBase
from ...layerlist import LayerList

LOGGER = logging.getLogger("napari.monitor")


class RemoteCommands:
    """Commands that a remote client can call.

    The MonitorApi commands a shared Queue calls "commands" that
    clients can put commands into.

    When MonitorApi.poll() is called, it checks the queue. It calls
    its run_command event for every command in the queue.

    This class listens to that event and processes those commands. The reason
    we use an event is so the monitor modules do not need to depend on
    Layer or LayerList. If they did it would create circular dependencies
    because people need to be able to import the monitor from anywhere.

    Parameters
    ----------
    layers : LayerList
        The viewer's layers, so we can call into them.

    Notes
    -----
    This is kind of a crude system for mapping remote commands to local methods,
    there probably is a better way with fancier use of events or something else.
    Also long term we don't what this to become a centralized repository of
    commands, command implementations should be spread out all over the system.
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
            if isinstance(layer, _OctreeImageBase):
                layer.display.show_grid = show

    def process_command(self, event) -> None:
        """Process this one command from the remote client.

        Parameters
        ----------
        event : dict
            The remote command.
        """
        command = event.command
        LOGGER.info("RemoveCommands._process_command: %s", json.dumps(command))

        # Every top-level key in in the command should be a method
        # in this RemoveCommands class.
        #
        #     { "set_grid": True }
        #
        # Then we would call self.set_grid(True)
        #
        for name, args in command.items():
            try:
                method = getattr(self, name)
                LOGGER.info("Calling RemoteCommands.%s(%s)", name, args)
                method(args)
            except AttributeError:
                LOGGER.error("RemoteCommands.%s does not exist.", name)
