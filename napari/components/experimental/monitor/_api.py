"""MonitorApi class.
"""
import logging
from multiprocessing.managers import SharedMemoryManager
from queue import Empty, Queue
from threading import Event
from typing import NamedTuple

from ....utils.events import EmitterGroup

LOGGER = logging.getLogger("napari.monitor")


class NapariRemoteAPI(NamedTuple):
    """Napari exposes these shared resources."""

    napari_shutting_down: Event
    commands: Queue
    client_messages: Queue
    data: dict


class MonitorApi:
    """The API that monitor clients can access.

    MonitorApi will execute commands from the clients when it is polled.

    Shared Resources
    ----------------
    Clients can access these shared resources via their SharedMemoryManager
    that connects to napari.

    shutdown : Event
        Signaled when napari is shutting down.

    commands : Queue
        Client can put "commands" on this queue.

    client_messages : Queue
        Clients receive messages on this queue.

    data : dict
        Generic data from monitor.add()

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

    # Is there a better way to do this? These can't be an attribute of
    # MonitorApi or the manager will try to pickle them. These instances
    # are NOT updated. Only the proxies returned by the manager are.
    #
    # And we need callables that return them, but lambda's don't work
    # because then THEY need to be pickled. So have these silly
    # staticmethods right now.
    _event = Event()
    _command_queue = Queue()
    _messages_queue = Queue()
    _dict = dict()

    @staticmethod
    def _get_event() -> Event:
        return MonitorApi._event

    @staticmethod
    def _get_command_queue() -> Queue:
        return MonitorApi._command_queue

    @staticmethod
    def _get_messages_queue() -> Queue:
        return MonitorApi._messages_queue

    @staticmethod
    def _get_dict() -> dict:
        return MonitorApi._dict

    def __init__(self):
        # We expose the run_command event so RemoteCommands can hook to it,
        # so it can execute commands we receive from clients.
        self.events = EmitterGroup(
            source=self, auto_connect=True, run_command=None
        )

        # We need to register all our callbacks before we create our
        # instance of SharedMemoryManager. Callbacks generally return
        # objects that SyncManager can create proxy objects for.
        SharedMemoryManager.register(
            'napari_shutting_down', callable=self._get_event
        )
        SharedMemoryManager.register(
            'commands', callable=self._get_command_queue
        )
        SharedMemoryManager.register(
            'client_messages', callable=self._get_messages_queue
        )
        SharedMemoryManager.register('data', callable=self._get_dict)

        # We ask for port 0 which means let the OS choose a port. We send
        # the chosen port to the client in its NAPARI_MON_CLIENT variable.
        self._manager = SharedMemoryManager(
            address=('127.0.0.1', 0), authkey=str.encode('napari')
        )
        self._manager.start()

        # Get the shared resources.
        self._remote = NapariRemoteAPI(
            self._manager.napari_shutting_down(),
            self._manager.commands(),
            self._manager.client_messages(),
            self._manager.data(),
        )

    @property
    def manager(self) -> SharedMemoryManager:
        """Our shared memory manager."""
        return self._manager

    def stop(self) -> None:
        """Notify clients we are shutting down."""
        self._remote.napari_shutting_down.set()

    def poll(self):
        """Poll the MonitorApi for new commands, etc."""
        assert self._manager is not None
        self._process_commands()

    def _process_commands(self) -> None:
        """Process every new command in the queue."""

        while True:
            try:
                command = self._remote.commands.get_nowait()

                if not isinstance(command, dict):
                    LOGGER.warning("Command was not a dict: %s", command)
                    continue

                self.events.run_command(command=command)
            except Empty:
                return  # No commands to process.

    def add(self, data: dict) -> None:
        """Add data for shared memory clients to read.

        Parameters
        ----------
        data : dict
            Add this data, replacing anything with the same key.
        """
        self._remote.data.update(data)

    def post(self, message: dict) -> None:
        """Post a message to shared memory clients."""
        self._remote.client_messages.put(message)
