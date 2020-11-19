"""Monitor class.

Experimental shared memory monitor.

With this monitor a program can publish data to shared memory. It will
startup any number of clients and pass them a blob of JSON as
configuration. The blob will contain at minimum the name of some shared
memory resource.

The client can read out of the shared memory and do whatever it wants with
the data. A possible client is a Flask-SocketIO web server. The user can
point their web browser at some port, and get some sort of dynamic
visualization of what's going in inside the program.

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
        "shared_list_name": "<name>"
    }

But it will evolve over time.

Currently the only shared resources is a single ShareableList, the client
connects to the shared_list_name above like this:

    shared_list = ShareableList(name=list_name)

The list has just two entires with these indexes:

    SLOT_FRAME_NUMBER = 0
    SLOT_JSON_BLOB = 1

The client can check shared_list[SLOT_FRAME_NUMBER] for the integer frame
number. If it sees a new number, it can decode the string in
shared_list[SLOT_JSON_BLOB].

The blob will be the union of every call made to monitor.add_aded() inside
napari. For example within napari you can do:

    monitor.add_data({"frame_time": delta_seconds})

somewhere else you can

    data = {
        "tiled_image_layer": {
            "num_created": stats.created,
            "num_deleted": stats.deleted,
            "duration_ms": elapsed.duration_ms,
        }
    }
    monitor.add_data(data)

The client can look for data['frame_time'] or data['tiled_image_layer'] or
any nested value. The client should be resilient, so that it does not
crash if something missing.

In summary within napari you can call monitor.add_data() from anywhere.
Once per frame the data is unioned together and written to shared memory as
a string, then the frame number is incremented.

The time to encode the example data above and write it to shared memory was
measured at less then 0.1 milliseconds. But it would get slower as it got
bigger.

Future Work
-----------
Use numpy recarray for bulk binary data that's not appropriate for JSON. We
can keep the JSON blob slot for low-data clients, but add new slots or
completely separate shared memory buffers.
"""
import base64
import copy
import errno
import json
import os
import subprocess
import time
from multiprocessing.shared_memory import ShareableList
from pathlib import Path
from threading import Event, Thread

from ....utils.perf import block_timer


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
client_config_template = {"shared_list_name": "<name>"}

# Create empty string of this size up front, since cannot grow it.
BUFFER_SIZE = 1024 * 1024

# Slots in our ShareableList, this is probably not a good system, but
# it's an easy way to prototype.
SLOT_FRAME_NUMBER = 0
SLOT_JSON_BLOB = 1


def _load_config(config_path: str) -> dict:
    """Load the JSON formatted config file.

    Parameters
    ----------
    config_path : str
        The path of the JSON file we should load.

    Return
    ------
    dict
        The parsed data from the JSON file.
    """
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            errno.ENOENT, f"Napari Monitor config file not found: {path}"
        )

    with path.open() as infile:
        return json.load(infile)


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

    print(f"NapariMon: starting client {args}")

    # Use Popen to run and do not wait for it to finish.
    subprocess.Popen(args, env=env)


def _get_monitor_config():
    """Return the NapariMonitor config file data, or None.

    Return
    ------
    dict
        The parsed config file data.
    """
    value = os.getenv("NAPARI_MON")
    if value in [None, "0"]:
        return None
    return _load_config(value)


class MonitorService(Thread):
    """Make data available to a client via shared memory.

    We are using JSON via ShareableList for prototyping with small amounts
    of data. It looks like using numpy's recarray with complex fields is
    the most powerful approach. Definitely do not use JSON for data of
    any non-trivial size.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Anyone can add to data with our self.add_data() then once per
        # frame we encode it into JSON and write into the shared list.
        self.data = {}
        self.frame_number = 0

        # We create the shared list in the thread.
        self.shared_list = None
        self.shared_list_name = None

        # Start our thread.
        num_clients = len(self.config['clients'])
        print(f"Monitor: Starting with {num_clients} clients.")
        self.ready = Event()
        self.start()

        # Wait for shared memory to be setup then start clients.
        self.ready.wait()
        self._start_clients()
        print(f"Monitor: Started {num_clients} clients.")

    def _start_clients(self):
        """Start every client in our config."""

        # Every client gets the same config, stuff in current values.
        client_config = copy.deepcopy(client_config_template)
        client_config['shared_list_name'] = self.shared_list_name

        for args in self.config['clients']:
            _start_client(args, client_config)

    def start_clients(self):
        """Return once the shared memory is setup."""

    def run(self):
        """Setup shared memory and wait."""
        # Create placeholder so we reserve the space. Probably not the best
        # way to pass JSON but it works.
        place_holder = " " * BUFFER_SIZE

        self.shared_list = ShareableList([self.frame_number, place_holder])
        self.shared_list_name = self.shared_list.shm.name

        # Post the empty JSON or clients will choke on the blank string.
        self._post_data()

        # All good.
        self.ready.set()

        # We don't really need a thread right now, but maybe?
        time.sleep(10000000)

    def _post_data(self):
        """Encode data as JSON and write it to shared memory."""
        with block_timer("json encode", print_time=True):
            self.shared_list[SLOT_JSON_BLOB] = json.dumps(self.data)

    def add_data(self, data):
        """Add data, combined data will be posted once per frame."""
        self.data.update(data)
        self.frame_number += 1  # for now, need timer instead
        self.shared_list[SLOT_FRAME_NUMBER] = self.frame_number
        self._post_data()

    def get_shared_name(self):
        """Wait and then return the shared name."""
        self.ready.wait()
        return self.shared_name


class Monitor:
    """Wrapper so we only start the service if NAPARI_MON was defined.

    Any calls to monitor.add() will be no-ops if the service was not
    started.
    """

    def __init__(self):
        self.service = None
        self.data = _get_monitor_config()

    def start(self):
        """Start the monitor service if configured

        Only start if NAPARI_MON was defined and pointed to a JSON file
        that we were able to parse.
        """
        if self.data is not None:
            self.service = MonitorService(self.data)

    def add(self, data):
        """Add monitoring data."""
        if self.service is not None:
            self.service.add_data(data)


monitor = Monitor()
