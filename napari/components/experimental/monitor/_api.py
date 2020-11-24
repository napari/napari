"""Monitor API.
"""
import os
from multiprocessing.managers import SharedMemoryManager
from queue import Empty, Queue

from ...layerlist import LayerList
from ._commands import MonitorCommands


class MonitorApi:
    """The API that monitor clients can access.

    There is only one API command right now:
        command_queue

    The commands returneds a Queue() proxy object that the client can add
    "commands" to which we'll process when we are polled.

    The SharedMemoryManager provides the same proxy objects as SyncManager
    including Queue, but also dict, list and others. So we can add more
    commands with more/different types of objects.

    See the docs for multiprocessing.managers.SyncManager.

    Parameters
    ----------
    Layer : LayerList
        The viewer's layers.
    """

    # This can't be an attribute of MonitorApi or the manager will try to
    # pickle it. Note this instance does not get updated. Only the proxy
    # returned by manager.get_queue actually has items in it.
    _queue = Queue()

    def get_queue():
        """Can't use a lambda or manager will try to pickle it."""
        return MonitorApi._queue

    def __init__(self, layers: LayerList):
        self.commands = MonitorCommands(layers)

        # We need to register these before the instance of
        # SharedMemoryManager is create inside MonitorService.
        SharedMemoryManager.register(
            'command_queue', callable=MonitorApi.get_queue
        )

        # Set once a manager is assigned.
        self.command_queue = None

    def set_manager(self, manager: SharedMemoryManager) -> None:
        """Set the shared manager we will get values from.

        Parameters
        ----------
        manager : SharedMemoryManager
            Get values from this manager.
        """
        # Now that there is a manager, we can get the proxy object.
        self.command_queue = manager.command_queue()

    def poll(self):
        """Poll the shared dict for new commands."""
        if self.command_queue is None:
            return  # Nothing to poll yet.

        while True:
            try:
                command = self.command_queue.get_nowait()
                self._process_command(command)
            except Empty:
                return  # No commands to process.

    def _process_command(self, command: dict):
        for command, args in command.items():
            method = getattr(self.commands, command)
            method(args)

    def _log(self, msg: str) -> None:
        # print for now but change to logging
        print(f"MonitorApi: process={os.getpid()} {msg}")
