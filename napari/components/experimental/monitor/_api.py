"""MonitorApi class.
"""
import logging
from multiprocessing.managers import SharedMemoryManager
from queue import Empty, Queue
from threading import Event
from typing import NamedTuple

from ....utils.events import EmitterGroup

LOGGER = logging.getLogger("napari.monitor")

# The client needs to know this.
AUTH_KEY = "napari"

# Port 0 means the OS chooses an available port. We send the server_port
# port to the client in its NAPARI_MON_CLIENT variable.
SERVER_PORT = 0


class NapariRemoteAPI(NamedTuple):
    """Napari exposes these shared resources."""

    napari_data: dict
    napari_messages: Queue
    napari_shutdown: Event

    client_data: dict
    client_messages: Queue


class MonitorApi:
    """The API that monitor clients can access.

    The MonitorApi creates and exposes a few "shared resources" that
    monitor clients can access. Client access the shared resources through
    their SharedMemoryManager which connects to napari.

    Exactly what resources we should expose is TBD. Here we are
    experimenting with having queue for sending message in each direction,
    and a shared dict for sharing data in both directions.

    The advantage of a Queue is presumably the other party will definitely
    get the message. While the advantage of dict is kind of the opposite,
    the other party can check the dict if they want, or they can ignore it.

    Again we're not sure what's best yet. But this illustrates some options.

    Shared Resources
    ----------------
    napari_data : dict
        Napari shares data in this dict for clients to read.

    napari_messages : Queue
        Napari puts messages in here for clients to read.

    napari_shutdown : Event
        Napari signals this event when shutting down. Although today napari
        does not wait on anything, so typically the client just gets a
        connection error when napari goes away, rather than seeing this event.

    client_data : Queue
        Client shares data in here for napari to read.

    client_messages : Queue
        Client puts messages in here for napari to read, such as commands.

    Notes
    -----
    The SharedMemoryManager provides the same proxy objects as SyncManager
    including list, dict, Barrier, BoundedSemaphore, Condition, Event,
    Lock, Namespace, Queue, RLock, Semaphore, Array, Value.

    SharedMemoryManager is derived from BaseManager, but it has similar
    functionality to SyncManager. See the official Python docs for
    multiprocessing.managers.SyncManager.

    Numpy can natively use shared memory buffers, something we want to try.
    """

    # BaseManager.register() is a bit weird. Not sure now to best deal with
    # it. Most ways I tried led to pickling errors, because this class is being run
    # in the shared memory server process? Feel free to find a better approach.
    _napari_data_dict = dict()
    _napari_messages_queue = Queue()
    _napari_shutdown_event = Event()

    _client_data_dict = dict()
    _client_messages_queue = Queue()

    @staticmethod
    def _napari_data() -> Queue:
        return MonitorApi._napari_data_dict

    @staticmethod
    def _napari_messages() -> Queue:
        return MonitorApi._napari_messages_queue

    @staticmethod
    def _napari_shutdown() -> Event:
        return MonitorApi._napari_shutdown_event

    @staticmethod
    def _client_data() -> Queue:
        return MonitorApi._client_data_dict

    @staticmethod
    def _client_messages() -> Queue:
        return MonitorApi._client_messages_queue

    def __init__(self):
        # RemoteCommands listens to our run_command event. It executes
        # commands from the clients.
        self.events = EmitterGroup(source=self, run_command=None)

        # We must register all callbacks before we create our instance of
        # SharedMemoryManager. The client must do the same thing, but it
        # only needs to know the names. We allocate the shared memory.
        SharedMemoryManager.register('napari_data', callable=self._napari_data)
        SharedMemoryManager.register(
            'napari_messages', callable=self._napari_messages
        )
        SharedMemoryManager.register(
            'napari_shutdown', callable=self._napari_shutdown
        )
        SharedMemoryManager.register('client_data', callable=self._client_data)
        SharedMemoryManager.register(
            'client_messages', callable=self._client_messages
        )

        # Start our shared memory server.
        self._manager = SharedMemoryManager(
            address=('127.0.0.1', SERVER_PORT), authkey=str.encode(AUTH_KEY)
        )
        self._manager.start()

        # Get the shared resources the server created. Clients will access
        # these same resources.
        self._remote = NapariRemoteAPI(
            self._manager.napari_data(),
            self._manager.napari_messages(),
            self._manager.napari_shutdown(),
            self._manager.client_data(),
            self._manager.client_messages(),
        )

    @property
    def manager(self) -> SharedMemoryManager:
        """Our shared memory manager.

        The wrapper Monitor class accesses this and passes it to the
        MonitorService.

        Returns
        -------
        SharedMemoryManager
            The manager we created and are using.
        """
        return self._manager

    def stop(self) -> None:
        """Notify clients we are shutting down.

        If we wanted a graceful shutdown, we could wait on "connected"
        clients to exit. With a short timeout in case they are hung.

        Today we just signal this event and immediately exit. So most of
        the time clients just get a connection error. They never see that
        this event was set.
        """
        self._remote.napari_shutdown.set()

    def poll(self):
        """Poll client_messages for new messages."""
        assert self._manager is not None
        self._process_client_messages()

    def _process_client_messages(self) -> None:
        """Process every new message in the queue."""

        client_messages = self._remote.client_messages
        while True:
            try:
                message = client_messages.get_nowait()

                if not isinstance(message, dict):
                    LOGGER.warning(
                        "Ignore message that was not a dict: %s", message
                    )
                    continue

                # Assume every message is a command that napari should
                # execute. We might have other types of messages later.
                self.events.run_command(command=message)
            except Empty:
                return  # No commands to process.

    def add_napari_data(self, data: dict) -> None:
        """Add data for shared memory clients to read.

        Parameters
        ----------
        data : dict
            Add this data, replacing anything with the same key.
        """
        self._remote.napari_data.update(data)

    def send_napari_message(self, message: dict) -> None:
        """Send a message to shared memory clients.

        Parameters
        ----------
        message : dict
            Message to send to clients.
        """
        self._remote.napari_messages.put(message)
