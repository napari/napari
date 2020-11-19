"""Monitor class.

Experimental shared memory monitor/server.

Run napari with NAPARI_MON=~/.mon
Where .mon is a JSON file like:

{
    "clients": [
        ["python", "/tmp/myclient.py"]
    ]
}

Monitor will run all given clients on start. The client should check
environment variable NAPARI_MON_CLIENT. It will contain a base64 encoded
string which it can parse as JSON like this:

def _get_client_config() -> dict:
    env_str = os.getenv("NAPARI_MON_CLIENT")
    if env_str is None:
        return None

    env_bytes = env_str.encode('ascii')
    config_bytes = base64.b64decode(env_bytes)
    config_str = config_bytes.decode('ascii')

    return json.loads(config_str)

For now the client config is like this:

{
    "shared_list_name": "<name>"
}

But it will evolve rapidly to start.
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


class Monitor(Thread):
    """Make data available to a client via shared memory.

    We are using JSON via ShareableList for prototyping with small amounts
    of data. It looks like using numpy's recarray with complex fields is
    the most powerful approach. Definitely do not use JSON for data of
    any non-trivial size.
    """

    def __init__(self, data):
        super().__init__()
        self.data = data

        # Create in the thread.
        self.shared_list = None
        self.shared_list_name = None

        # Start our thread.
        num_clients = len(self.data['clients'])
        print(f"NapariMon: starting with {num_clients}")
        self.ready = Event()
        self.start()

        # Wait for shared memory to be setup then start clients.
        self.ready.wait()
        self._start_clients()
        print(f"NapariMon: started {num_clients} clients")

    def _start_clients(self):
        """Start every client in our config."""

        # Every client gets the same config, stuff in current values.
        client_config = copy.deepcopy(client_config_template)
        client_config['shared_list_name'] = self.shared_list_name

        for args in self.data['clients']:
            _start_client(args, client_config)

    def start_clients(self):
        """Return once the shared memory is setup."""

    def run(self):
        """Setup shared memory and wait."""
        place_holder_str = " " * BUFFER_SIZE

        self.shared_list = ShareableList([place_holder_str])
        self.shared_list_name = self.shared_list.shm.name

        self.ready.set()
        time.sleep(10000000)

    def post_message(self, data):
        """Post a message to shared memory."""
        json_str = json.dumps(data)
        self.shared_list[0] = json_str

    def get_shared_name(self):
        """Wait and then return the shared name."""
        self.ready.wait()
        return self.shared_name


def _create_monitor():
    """Start the shared memory monitor."""
    data = _get_monitor_config()

    if data is None:
        return None  # Env var not set, do not create.

    return Monitor(data)


monitor = _create_monitor()
