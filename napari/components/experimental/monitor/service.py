"""Monitor class.

Experimental shared memory monitor.

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
        "shared_list_name": "<name>"
    }

But it will evolve over time.

Currently the only shared resource is a single ShareableList, the client
can connect to the list like this:

    shared_list = ShareableList(name=data['shared_list_name'])

The list has just two entries right now. With these indexes:

    FRAME_NUMBER = 0
    FROM_NAPARI = 1
    TO_NAPARI = 2

The client can check shared_list[FRAME_NUMBER] for the integer frame
number. If it sees a new number, it can decode the string in
shared_list[FROM_NAPARI].

The blob will be the union of every call made to monitor.add(). For example
within napari you can do:

    monitor.add_data({"frame_time": delta_seconds})

somewhere else you can do:

    data = {
        "tiled_image_layer": {
            "num_created": stats.created,
            "num_deleted": stats.deleted,
            "duration_ms": elapsed.duration_ms,
        }
    }
    monitor.add_data(data)

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
import subprocess
import time
from multiprocessing.managers import SharedMemoryManager
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

# Slots in our ShareableList, this is probably not a good system, but
# it's an easy way to prototype.
FRAME_NUMBER = 0
FROM_NAPARI = 1
TO_NAPARI = 2


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

    print(f"Monitor: starting client {args}")

    # Use Popen to run and do not wait for it to finish.
    subprocess.Popen(args, env=env)


def _test_callable():
    print("TEST CALLABLE")


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

        SharedMemoryManager.register('test_callable', callable=_test_callable)

        self.manager = SharedMemoryManager()
        self.manager.start()

        # Right now JSON from clients is written into here. Hopefully this
        # will go away if we start using the BaseManager callback feature.
        self.from_client = {}

        # Anyone can add to data with our self.add_data() then once per
        # frame we encode it into JSON and write into the shared list.
        self.data = {}
        self.frame_number = 0

        # We create the shared list in the thread.
        self.shared_list = None

        # Start our thread.
        num_clients = len(self.config['clients'])
        print(f"Monitor: Starting with {num_clients} clients.")
        self.ready = Event()
        self.start()

        # Wait for shared memory to be setup then start clients.
        self.ready.wait()

        # We could start clients from the thread, but that might lead to
        # confusing with prints/logging as napari and the clients are
        # starting up at the exact same time. Maybe someday.
        self._start_clients()
        print(f"Monitor: Started {num_clients} clients.")

    def _start_clients(self) -> None:
        """Start every client in our config."""

        # Every client gets the same config, stuff in current values.
        client_config = copy.deepcopy(client_config_template)
        client_config['shared_list_name'] = self.shared_list.shm.name

        for args in self.config['clients']:
            _start_client(args, client_config)

    def run(self) -> None:
        """Setup shared memory and wait."""
        # Create placeholder so we reserve the space. Probably not the best
        # way to pass JSON but it works. See shared memory with numpy
        # for much more powerful options or all types including strings.
        buffer_str = "{}" + " " * BUFFER_SIZE

        # These are our three shared memory slots.
        # FRAME_NUMBER = 0
        # FROM_NAPARI = 1
        # TO_NAPARI = 2
        slots = [self.frame_number, buffer_str, buffer_str]
        self.shared_list = self.manager.ShareableList(slots)

        # Poll once so slots have at least valid empty JSON.
        self.poll()

        # All good.
        self.ready.set()

        # We don't really need a thread right now, but maybe?
        time.sleep(10000000)

    def poll(self) -> None:
        """Shuttle data from/to napari."""
        # Post accumulated data from napari.
        self.shared_list[FROM_NAPARI] = json.dumps(self.data)

        # Get new data from clients.
        json_str = self.shared_list[TO_NAPARI].rstrip()
        if len(json_str) > 3:
            print(f"Monitor: from client: {json_str}")

        try:
            self.from_client = json.loads(json_str)
        except json.decoder.JSONDecodeError:
            print(f"Monitor: error parsing json: {json_str}")

        if self.from_client:
            print(f"Monitor: data from clients: {json_str}")

        # Update the frame number so clients know something changed.
        # Better would be a queue they can wait on?
        self.frame_number += 1
        self.shared_list[FRAME_NUMBER] = self.frame_number

    def add_data(self, data) -> None:
        """Add data, combined data will be posted once per frame."""
        self.data.update(data)

    def get_shared_name(self) -> None:
        """Wait and then return the shared name."""
        self.ready.wait()
        return self.shared_name

    def stop(self) -> None:
        """Stop the shared memory service."""
        self.manager.shutdown()
