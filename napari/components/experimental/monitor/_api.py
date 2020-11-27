"""MonitorApi class.
"""
import logging
import os
from multiprocessing.managers import SharedMemoryManager
from queue import Empty, Queue
from threading import Event

from ...layerlist import LayerList
from ._commands import MonitorCommands

LOGGER = logging.getLogger("napari.monitor")


class MonitorApi:
    """The API that monitor clients can access.

    We expose these API commands to the client:
        shutdown_event
        command_queue

    shutdown_event -> Event()
        Signaled when napari is shutting down.

    command_queue -> Queue()
        Client can put "commands" on this queue.

    MonitorApi will execute the commands when it is polled.

    Notes
    -----
    The SharedMemoryManager provides the same proxy objects as SyncManager
    including list, dict, Barrier, BoundedSemaphore, Condition, Event,
    Lock, Namespace, Queue, RLock, Semaphore, Array, Value.

    SharedMemoryManager is derived from BaseManager, but it has similar
    functionality to SyncManager. See the docs for
    multiprocessing.managers.SyncManager.

    Parameters
    ----------
    Layer : LayerList
        The viewer's layers.
    """

    # These can't be an attribute of MonitorApi or the manager will try to
    # pickle them. These instances are NOT updated. Only the proxies returned
    # by the manager are.
    _event = Event()
    _queue = Queue()
    _dict = dict()

    @staticmethod
    def _get_event() -> Event:
        return MonitorApi._event

    @staticmethod
    def _get_queue() -> Queue:
        return MonitorApi._queue

    @staticmethod
    def _get_dict() -> dict:
        return MonitorApi._dict

    def __init__(self, layers: LayerList):
        # We expect there's a MonitorCommands method for every command
        # that we pull out of the queue.
        self._commands = MonitorCommands(layers)
        self._pid = os.getpid()

        # We need to register all our callbacks before we create our
        # instance of SharedMemoryManager. Callbacks generally return
        # objects that SyncManager can create proxy objects for.
        SharedMemoryManager.register(
            'shutdown_event', callable=self._get_event
        )
        SharedMemoryManager.register('command_queue', callable=self._get_queue)
        SharedMemoryManager.register('data', callable=self._get_dict)

        # We ask for port 0 which means let the OS choose a port. We send
        # the chosen port to the client in its NAPARI_MON_CLIENT variable.
        self._manager = SharedMemoryManager(
            address=('127.0.0.1', 0), authkey=str.encode('napari')
        )
        self._manager.start()

        # Now we have these proxy objects.
        self._shutdown_event = self._manager.shutdown_event()
        self._command_queue = self._manager.command_queue()
        self._data = self._manager.data()

    @property
    def manager(self) -> SharedMemoryManager:
        """Our shared memory manager."""
        return self._manager

    def stop(self) -> None:
        """Notify clients we are shutting down."""
        self._shutdown_event.set()

    def poll(self):
        """Poll the MonitorApi for new commands, etc."""
        assert self._manager is not None
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
        # Then we would call self._commands.set_grid(True)
        #
        for name, args in command.items():
            try:
                method = getattr(self._commands, name)
                LOGGER.info("Calling MonitorCommands.%s(%s)", name, args)
                method(args)
            except AttributeError:
                LOGGER.error("MonitorCommands.%s does not exist.", name)

    def add_data(self, data: dict) -> None:
        """Add data for shared memory clients to read.

        Parameters
        ----------
        data : dict
            Add this data, replacing anything with the same key.
        """
        self._data.update(data)
