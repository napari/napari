"""Monitor class.

Monitor is a wrapper for our MonitorService.
"""
import errno
import json
import os
import sys
from pathlib import Path
from typing import Optional


def _create_monitor_service(config: dict):
    """Create our MonitorServer if possible.

    For now Python 3.8 is required since it introduced some new shared
    memory features. The new features have been called "real shared
    memory". The old shared memory features were implemented with sockets?
    The new way is supposedly MUCH faster.

    Possibly there is a way to get this working with Python 3.7. Maybe
    under 3.7 we only support small-ish JSON data, and not full binary
    buffers.

    Parameters
    ----------
    config : dict
        Configuration for the service.
    """
    if sys.version_info[:2] >= (3, 8):
        from .service import MonitorService

        return MonitorService(config)

    return None


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
            errno.ENOENT, f"Monitor: Config file not found: {path}"
        )

    with path.open() as infile:
        return json.load(infile)


def _get_monitor_config() -> Optional[dict]:
    """Return the MonitorService config file data, or None.

    Return
    ------
    Optional[dict]
        The parsed config file data.
    """
    value = os.getenv("NAPARI_MON")
    if value in [None, "0"]:
        return None
    return _load_config(value)


class Monitor:
    """Wrapper around our MonitorService.

    Notes
    -----
    We have this wrapper so we only start the service if NAPARI_MON was
    defined and we have Python 3.8 or newer.

    Also having a wrapper means callers do not have to check if the monitor
    exists. The can just call monitor.foo() regardless.
    """

    def __init__(self):
        self.service = None

        # Only if NAPARI_MON was defined and pointed to a JSON file that we
        # were able to parse will we have a config.
        self.config = _get_monitor_config()

    def start(self):
        """Start the monitor service, if possible.

        We could create the service in our __init__ but we might not want
        it starting up at import time.
        """
        if self.config is not None:
            # The create will return None if we don't meet the requirements.
            self.service = _create_monitor_service(self.config)

    def stop(self):
        """TODO_MON: call this somewhere!"""
        if self.service is not None:
            self.service.stop()

    def add(self, data):
        """Add monitoring data."""
        if self.service is not None:
            self.service.add_data(data)

    def poll(self):
        """Someone must poll the service once per frame."""
        if self.service is not None:
            self.service.poll()


monitor = Monitor()
