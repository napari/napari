"""Global monitor server.

The global "monitor" object will be None unless NAPARI_MON points to a
parsesable config file and we are running under Python 3.9 or newer.
"""
import errno
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ...layerlist import LayerList

if TYPE_CHECKING:
    from ._service import MonitorService

# If False monitor is disabled even if we meet all other requirements.
ENABLE_MONITOR = True


def _load_config(path: str) -> dict:
    """Load the JSON formatted config file.

    Parameters
    ----------
    path : str
        The path of the JSON file we should load.

    Return
    ------
    dict
        The parsed data from the JSON file.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            errno.ENOENT, f"Monitor: Config file not found: {path}"
        )

    with path.open() as infile:
        return json.load(infile)


def _load_monitor_config() -> Optional[dict]:
    """Return the MonitorService config file data, or None.

    Return
    ------
    Optional[dict]
        The parsed config file data or None if no config.
    """
    # We shouldn't even call into this file unless NAPARI_MON is defined
    # but check to be sure.
    value = os.getenv("NAPARI_MON")
    if value in [None, "0"]:
        return None

    return _load_config(value)


def _get_monitor_config() -> Optional[dict]:
    """Create and return the configuration for the MonitorService.

    The routine might return None for one serveral reasons:
    1) We're not running under Python 3.9 or now.
    2) The monitor is explicitly disable, ENABLED_MONITOR is False.
    3) The NAPARI_MON environment variable is not defined.
    4) The NAPARI_MON config file cannot be found and parsed.

    Return
    ------
    Optional[dict]
        The configuration for the MonitorService.
    """
    if sys.version_info[:2] < (3, 9):
        # We require Python 3.9 for now. The shared memory features we need
        # were added in 3.8, but the 3.8 implemention was buggy. It's
        # possible we could backport to or otherwise fix 3.8 or even 3.7,
        # but for now we're making 3.9 a requirement.
        print("Monitor: not starting, requires Python 3.9 or newer")
        return None

    if not ENABLE_MONITOR:
        print("Monitor: not starting, disabled")
        return None

    # The NAPARI_MON environment variable points to our config file.
    config = _load_monitor_config()

    if config is None:
        print("Monitor: not starting, no usable config file")
        return None

    return config


class Monitor:
    """Wraps the monitor service.

    We can't start the monitor service at import time. Under the hood the
    multiprocessing complains about a "partially started process".

    Instead someone must call our start() method explicitly once the
    process has fully started.
    """

    def __init__(self):
        # Both are set when start() is called, and only if we have
        # a parseable config file, have Python 3.9, etc.
        self._service = None
        self._api = None

    def __nonzero__(self):
        """Return True if the service is running.

        So that callers can do:

            if monitor:
                monitor.add(...)
        """
        return self._service is not None

    def start(self, layers: LayerList) -> bool:
        """Start the monitor service, if it hasn't been started already.

        Return
        ------
        bool
            True if we started the service or it was already started.
        """
        if self._service is not None:
            return True  # It was already started.

        config = _get_monitor_config()

        if config is None:
            return False  # Can't start without config.

        # Late imports so no multiprocessing modules are even
        # imported unless we are going to start the service.
        from ._api import MonitorApi
        from ._service import MonitorService

        # Create the API first so it can register our callbacks.
        self._api = MonitorApi(layers)
        self._service = MonitorService(config)

        # API needs the manager to fetch shared data.
        self._api.set_manager(self._service.manager)
        return True  # We started the service.

    def get(self) -> 'Optional[MonitorService]':
        """Return the MonitorService instance if it was started."""
        return self._service

    def poll(self):
        """Poll the monitor service if it was started."""
        if self._service is not None:
            self._service.poll()
            self._api.poll()

    def add(self, data):
        """Add data to the monitor service.

        Caller should use this pattern:
            if monitor:
                monitor.add(...)

        If they do not want to waste time create the input dict unless
        the service is running.
        """
        if self._service is not None:
            self._service.add_data(data)


monitor = Monitor()
