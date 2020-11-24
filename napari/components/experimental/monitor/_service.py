"""MonitorService class.

Experimental shared memory monitor service.

With this monitor a program can publish data to shared memory. The monitor
has a JSON config file. When the monitor starts will launch any number of
clients listed in that config file.

It will pass each client a blob of JSON as configuration data. The blob
will contain at minimum the name of some shared memory resource.

The client can read out of the shared memory and do whatever it wants with
the data. A possible client might be a Flask-SocketIO web server. The user
can point their web browser at some port and see a dynamic visualization of
what's going in inside the program.

Only if NAPARI_MON is set and points to a config file will the monitor even
start. Run napari like:

    NAPARI_MON=~/.napari-mon napari

The format of the .napari-mon config file is:

{
    "clients": [
        ["python", "/tmp/myclient.py"]
    ]
}

All of the listed clients will be started. They can be the same program run
with different arguments, or different programs. All clients will have
access to the same shared memory.

The client gets the shared memory information by checking for the
NAPARI_MON_CLIENT variable. It can be decoded like this:

    def _get_client_config() -> dict:
        env_str = os.getenv("NAPARI_MON_CLIENT")
        if env_str is None:
            return None

        env_bytes = env_str.encode('ascii')
        config_bytes = base64.b64decode(env_bytes)
        config_str = config_bytes.decode('ascii')

        return json.loads(config_str)

For now the client configuration is just:

    {
        "shared_list_name": "<name>",
        "server_port": "<number>"
    }

But it will evolve over time.

The list name refers to a ShareableList, the client can connect to it like
this:

    shared_list = ShareableList(name=data['shared_list_name'])

The server port should be used when creating a SharedMemoryManager:

    self.manager = SharedMemoryManager(
        address=('localhost', config['server_port']),
        authkey=str.encode('napari')
    )

The list has just two entries right now. With these indexes:

    FRAME_NUMBER = 0
    FROM_NAPARI = 1

The client can check shared_list[FRAME_NUMBER] for the integer frame
number. If it sees a new number, it can decode the string in
shared_list[FROM_NAPARI].

The blob will be the union of every call made to monitor.add(). For example
within napari you can do:

    monitor.add({"frame_time": delta_seconds})

somewhere else you can do:

    data = {
        "tiled_image_layer": {
            "num_created": stats.created,
            "num_deleted": stats.deleted,
            "duration_ms": elapsed.duration_ms,
        }
    }
    monitor.add(data)

Client can access data['frame_time'] or data['tiled_image_layer']. Clients
should be resilient, so that it does not crash if something missing.

In summary within napari you can call monitor.add() from anywhere. Once per
frame the combined data is written to shared memory as JSON, then the frame
number is incremented.

It only takes about 0.1 milliseconds to encode the above data and write to
shared memory. But this will slow down as more data is written.

Future Work
-----------
JSON is no appropriate for lots of data. A better solution is numpy/recarry
where it should be possible to have any number of fields of any time of
binary data. We can keep the JSON blob slot for ad-hoc use, or we can
replace it entirely with numpy. It was just simple to start with JSON
and ShareableList.
"""
import base64
import copy
import json
import os
import subprocess
from multiprocessing.managers import SharedMemoryManager

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
FROM_NAPARI = 1


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
        self._pid = os.getpid()

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
        self._log("listening on port {self.server_port}")

        num_clients = len(self._config['clients'])
        self._log(f"starting {num_clients} clients.")

        # Every client gets the same config, stuff in current values.
        client_config = copy.deepcopy(client_config_template)
        client_config['shared_list_name'] = self.shared_list.shm.name
        client_config['server_port'] = server_port

        # Start every client.
        for args in self._config['clients']:
            self._log(f"starting client {args}")
            _start_client(args, client_config)

        self._log(f"started {num_clients} clients.")

    def _create_shared_list(self) -> None:
        """Create our shared list."""
        # Create placeholder so we reserve the space. Probably not the best
        # way to pass JSON but it works. See shared memory with numpy
        # for much more powerful options or all types including strings.
        buffer_str = "{}" + " " * BUFFER_SIZE

        # These are our two shared memory slots.
        # FRAME_NUMBER = 0
        # FROM_NAPARI = 1
        slots = [self.frame_number, buffer_str, buffer_str]
        shared_list = self._manager.ShareableList(slots)

        return shared_list

    def poll(self) -> None:
        """Write accumulated data into shared memory."""
        self.shared_list[FROM_NAPARI] = json.dumps(self._data)

        # Update the frame number so clients know something changed.
        self.frame_number += 1
        self.shared_list[FRAME_NUMBER] = self.frame_number

    def add_data(self, data) -> None:
        """Add data, combined data will be posted once per frame."""
        self._data.update(data)

    def stop(self) -> None:
        """Stop the shared memory service."""
        self._manager.shutdown()

    def _log(self, msg: str) -> None:
        """Log a message.

        This is a print for now. But we should switch to logging.

        Parameters
        ----------
        msg : str
            The message to log.
        """
        print(f"MonitorService: process={self._pid} {msg}")
