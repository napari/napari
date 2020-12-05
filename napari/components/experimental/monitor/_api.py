"""MonitorApi class.
"""
import json
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

    # BaseManager.register() is a bit weird. There must be a better wa
    # to do this. But most things including lambda result in pickling
    # errors due to multiprocessing stuff.
    #
    # So we create these and then use staticmethods as our callables.
    _napari_shutting_down_event = Event()
    _commands_queue = Queue()
    _client_messages_queue = Queue()
    _data_dict = dict()

    @staticmethod
    def _napari_shutting_down() -> Event:
        return MonitorApi._napari_shutting_down_event

    @staticmethod
    def _commands() -> Queue:
        return MonitorApi._commands_queue

    @staticmethod
    def _client_messages() -> Queue:
        return MonitorApi._client_messages_queue

    @staticmethod
    def _data() -> dict:
        return MonitorApi._data_dict

    def __init__(self):
        # We expose the run_command event so RemoteCommands can hook to it,
        # so it can execute commands we receive from clients.
        self.events = EmitterGroup(
            source=self, auto_connect=True, run_command=None
        )

        # Must register all callbacks before we create our instance of
        # SharedMemoryManager.
        SharedMemoryManager.register(
            'napari_shutting_down', callable=self._napari_shutting_down
        )
        SharedMemoryManager.register('commands', callable=self._commands)
        SharedMemoryManager.register(
            'client_messages', callable=self._client_messages
        )
        SharedMemoryManager.register('data', callable=self._data)

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

    def send(self, message: dict) -> None:
        """Send a message to shared memory clients."""
        LOGGER.info("MonitorApi.send: %s", json.dumps(message))
        self._remote.client_messages.put(message)
