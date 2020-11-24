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

    @staticmethod
    def _get_queue():
        """Can't use a lambda or manager will try to pickle it."""
        return MonitorApi._queue

    def __init__(self, layers: LayerList):
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
                    self._log("ignoring non-dict command {command}")
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
        # Every top-level key in the dict should be a method in our
        # MonitorCommands class.
        for name, args in command.items():
            try:
                method = getattr(self._commands, name)
                method(args)
            except AttributeError:
                self._log(f"command not found {command}")

    def _log(self, msg: str) -> None:
        """Log a message.

        This is a print for now. But we should switch to logging.

        Parameters
        ----------
        msg : str
            The message to log.
        """
        print(f"MonitorApi: process={self._pid} {msg}")
