"""MonitorApi class.
"""
import logging
import os
from multiprocessing.managers import SharedMemoryManager
from queue import Empty, Queue

from ...layerlist import LayerList
from ._commands import MonitorCommands

LOGGER = logging.getLogger("napari.monitor")


class MonitorApi:
    """The API that monitor clients can access.

    There is only one API command right now:
        command_queue

    The command_queue function returneds a Queue() proxy object that the
    client can add "commands". We process these commands when polled.

    The SharedMemoryManager provides the same proxy objects as SyncManager
    including Queue, dict, list and many others. So we can add other
    API's that return other shared resources.

    See the docs for multiprocessing.managers.SyncManager.

    Parameters
    ----------
    Layer : LayerList
        The viewer's layers.
    """

    # This can't be an attribute of MonitorApi or the manager will try to
    # pickle it. Note this instance does not get updated. Only the proxy
    # returned by by the manager has items in it.
    _queue = Queue()

    @staticmethod
    def _get_queue() -> Queue:
        """Static method since can't use lambda, it can't be pickled."""
        return MonitorApi._queue

    def __init__(self, layers: LayerList):
        # We expect there's a MonitorCommands method for every command
        # that we pull out of the queue.
        self._commands = MonitorCommands(layers)
        self._pid = os.getpid()

        # We need to register all our callbacks before we create our
        # instance of SharedMemoryManager. Callbacks generally return
        # objects that SyncManager can create proxy objects for.
        #
        # SharedMemoryManager is derived from BaseManager, but it has
        # similar functionality to SyncManager.
        SharedMemoryManager.register('command_queue', callable=self._get_queue)

        # These are set in our start_manager().
        self._manager = None
        self._command_queue = None

    def start_manager(self) -> SharedMemoryManager:
        """Start our SharedMemoryManager and return it.

        Return
        ------
        SharedMemoryManager
            Our manager.
        """
        self._manager = SharedMemoryManager(
            address=('127.0.0.1', 0), authkey=str.encode('napari')
        )
        self._manager.start()

        # Now we have queue.
        self._command_queue = self._manager.command_queue()

        return self._manager

    def poll(self):
        """Poll the MonitorApi for new commands, etc."""
        if self._command_queue is None:
            return  # Nothing to poll yet.

        self._process_commands()

    def _process_commands(self) -> None:
        """Process every new command in the queue."""

        while True:
            try:
                command = self._command_queue.get_nowait()

                if not isinstance(command, dict):
                    LOGGER.warning("Command was not a dict: %s", command)
                    continue

                self._process_command(command)
            except Empty:
                return  # No commands to process.

    def _process_command(self, command: dict):
        """Process this one command from the remote client.

        Parameters
        ----------
        command : dict
            The remote command.
        """
        LOGGER.info("Processing command: %s", command)

        # Every top-level key in the dict should be a method in our
        # MonitorCommands class. For example if the dict is:
        #
        #     { "set_grid": True }
        #
        # Then we'll call self._commands.set_grid(True)
        #
        for name, args in command.items():
            try:
                method = getattr(self._commands, name)
                LOGGER.info("Calling MonitorCommands.%s(%s)", name, args)
                method(args)
            except AttributeError:
                LOGGER.error("MonitorCommands.%s does not exist.", name)
