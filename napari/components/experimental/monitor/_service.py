"""MonitorService class.

Experimental shared memory service.

1) Creates a ShareableList with two slots:
    FRAME_NUMBER = 0
    JSON_DATA = 1

2) Starts all clients specified in the config file.

3) Anywhere in napari can call MonitorService.add_data any number of times.

4) When MonitorService.poll() is called, the *union* of all the data is
   writted to slot JSON_DATA. JSON string. And the frame number is
   incremented in slot FRAME_NUMBER.

When a shared memory client sees the frame number has incremented, it can
grab the latest JSON from JSON_DATA.

Monitor Config File
-------------------
Only if NAPARI_MON is set and points to a config file will the monitor even
start. The format of the .napari-mon config file is:

{
    "clients": [
        ["python", "/tmp/myclient.py"]
    ]
    "log_path": "/tmp/monitor.log"
}

All of the listed clients will be started. They can be the same program run
with different arguments, or different programs. All clients will have
access to the same shared memory.

Client Config File
-------------------
The client should decode the contents of the NAPARI_MON_CLIENT variable. It
can be decoded like this:

    def _get_client_config() -> dict:
        env_str = os.getenv("NAPARI_MON_CLIENT")
        if env_str is None:
            return None

        env_bytes = env_str.encode('ascii')
        config_bytes = base64.b64decode(env_bytes)
        config_str = config_bytes.decode('ascii')

        return json.loads(config_str)

The client configuration is:

    {
        "shared_list_name": "<name>",
        "server_port": "<number>"
    }

The client can access the ShareableList like this:

    shared_list = ShareableList(name=data['shared_list_name'])

The client can access the MonitorApi by creating a SharedMemoryManager:

    SharedMemoryManager.register('command_queue')
    self.manager = SharedMemoryManager(
        address=('localhost', config['server_port']),
        authkey=str.encode('napari')
    )
    self._commands = self._manager.command_queue()

It can send command like:

    self._commands.put(
       {"test_command": {"value": 42, "names": ["fred", "joe"]}}
    )

Passing Data
------------
In napari add data like:

    if monitor:
        monitor.add({
            "tiled_image_layer": {
                "num_created": stats.created,
                "num_deleted": stats.deleted,
                "duration_ms": elapsed.duration_ms,
            }
        })

The client can access data['tiled_image_layer']. Clients should be
resilient if data is missing, in case the napari version is different than
expected.

Future Work
-----------
We plan to use numpy shared memory buffers /w recarray for bulk binary
data. JSON is not appropriate for bulk data, but it was simple thing
to start with. And is pretty fast given it's in shared memory.

The JSON_DATA might be replaced with a Queue or something as well.
"""
import base64
import copy
import json
import logging
import subprocess
from multiprocessing.managers import SharedMemoryManager

from ._utils import numpy_dumps

LOGGER = logging.getLogger("napari.monitor")

# If False we don't start any clients, for debugging.
START_CLIENTS = True


def _base64_json(data: dict) -> str:
    """Return base64 encoded version of this data as JSON.

    data : dict
        The data to write as JSON then base64 encode.
    """
    json_str = json.dumps(data)
    json_bytes = json_str.encode('ascii')
    message_bytes = base64.b64encode(json_bytes)
    return message_bytes.decode('ascii')


# We send this to the client when it starts. So it can attach to our shared
# memory among other things. We insert the real <name>.
client_config_template = {
    "shared_list_name": "<name>",
    "server_port": "<number>",
}

# Create empty string of this size up front, since cannot grow it.
BUFFER_SIZE = 1024 * 1024

# Slots in our ShareableList, this is probably not a good system, but
# it's an easy way to prototype.
FRAME_NUMBER = 0
JSON_DATA = 1


def _start_client(args, client_config) -> None:
    """Start this one client, pass the config as an env variable.

    Parameters
    ----------
    args : List[str]
        The path of the client and any arguments.
    client_config : dict
        The data to pass the client.
    """
    env = {"NAPARI_MON_CLIENT": _base64_json(client_config)}

    # Use Popen to run and do not wait for it to finish.
    subprocess.Popen(args, env=env)


class MonitorService:
    """Make data available to a client via shared memory.

    We are using JSON via ShareableList for prototyping with small amounts
    of data. For bulk binary data, like images, it looks like using numpy's
    recarray with complex fields is the most powerful approach. Definitely
    do not use JSON for "lots" of data.
    """

    def __init__(self, config: dict, manager: SharedMemoryManager):
        super().__init__()
        self._config = config
        self._manager = manager

        # Anyone can add to data with our self.add_data() then once per
        # frame we encode it into JSON and write into the shared list.
        #
        # Using JSON and a shared list slot might be replaced by
        # SyncManager objects. But it was a quick way to get started.
        # And quite fast really.
        self._data = {}
        self.frame_number = 0

        # We expect to have more shared memory resources soon, but right
        # now it's just this shared list.
        self.shared_list = self._create_shared_list()

        if START_CLIENTS:
            self._start_clients()

    def _start_clients(self) -> None:
        """Start every client in our config."""
        # We asked for port 0 which means the OS will pick a port, we
        # save it off so we can send it the clients are starting up.
        server_port = self._manager.address[1]
        LOGGER.info("Listening on port %s", server_port)

        num_clients = len(self._config['clients'])
        LOGGER.info("Starting %d clients...", num_clients)

        # Every client gets the same config, stuff in current values.
        client_config = copy.deepcopy(client_config_template)
        client_config['shared_list_name'] = self.shared_list.shm.name
        client_config['server_port'] = server_port

        # Start every client.
        for args in self._config['clients']:
            LOGGER.info("Starting client %s", args)
            _start_client(args, client_config)

        LOGGER.info("Started %d clients.", num_clients)

    def _create_shared_list(self) -> None:
        """Create our shared list."""
        # Create placeholder so we reserve the space. Probably not the best
        # way to pass JSON but it works. See shared memory with numpy
        # for much more powerful options or all types including strings.
        buffer_str = "{}" + " " * BUFFER_SIZE

        # These are our two shared memory slots.
        # FRAME_NUMBER = 0
        # JSON_DATA = 1
        slots = [self.frame_number, buffer_str, buffer_str]
        shared_list = self._manager.ShareableList(slots)

        return shared_list

    def poll(self) -> None:
        """Write accumulated data into shared memory."""
        self.shared_list[JSON_DATA] = numpy_dumps(self._data)

        # Increment the frame number so clients know something changed.
        self.frame_number += 1
        self.shared_list[FRAME_NUMBER] = self.frame_number

    def add_data(self, data) -> None:
        """Add data, combined data will be posted once per frame."""
        self._data.update(data)

    def stop(self) -> None:
        """Stop the shared memory service."""
        self._manager.shutdown()
