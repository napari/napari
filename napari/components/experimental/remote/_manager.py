"""RemoteManager class.
"""
import logging

from napari.components.experimental.remote._commands import RemoteCommands
from napari.components.experimental.remote._messages import RemoteMessages
from napari.components.layerlist import LayerList
from napari.utils.events import Event

LOGGER = logging.getLogger("napari.monitor")


class RemoteManager:
    """Interacts with remote clients.

    The monitor system itself purposely does not depend on anything else in
    napari except for utils.events.

    However RemoteManager and its children RemoteCommands and
    RemoteMessages do very much depend on napari. RemoteCommands executes
    commands sent to napari by clients. RemoteMessages sends messages to
    remote clients, such as the current state of the layers.

    Parameters
    ----------
    layers : LayerList
        The viewer's layers.
    """

    def __init__(self, layers: LayerList):
        self._commands = RemoteCommands(layers)
        self._messages = RemoteMessages(layers)

    def process_command(self, event: Event) -> None:
        """Process this command from a remote client.

        Parameters
        ----------
        event : Event
            Contains the command to process.
        """
        return self._commands.process_command(event)

    def on_poll(self, _event: Event) -> None:
        """Send out messages when polled."""
        self._messages.on_poll()
