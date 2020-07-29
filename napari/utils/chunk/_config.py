"""AsyncConfig to configure asynchronous loading and the ChunkLoader.
"""
import errno
import json
import logging
import os
from pathlib import Path

LOGGER = logging.getLogger("ChunkLoader")

# If NAPARI_ASYNC=1 or NAPARI_ASYNC=/path/to/config.json then async
# loading is enabled. Otherwise it's disabled.
ASYNC_ENV_VAR = "NAPARI_ASYNC"

# NAPARI_ASYNC=0 or missing will use these settings:
DEFAULT_SYNC_CONFIG = {"synchronous": True}

# NAPARI_ASYNC=1 will use these settings:
DEFAULT_ASYNC_CONFIG = {
    "synchronous": False,
    "num_workers": 6,
    "log_path": None,
    "use_procesess": False,
    "delay_seconds": 0.1,
    "load_seconds": 0,
}


def _log_to_file(path: str) -> None:
    """Log ChunkLoader log messages to the given file path.

    Parameters
    ----------
    path : str
        Log to this file path.
    """
    if path:
        fh = logging.FileHandler(path)
        LOGGER.addHandler(fh)
        LOGGER.setLevel(logging.INFO)


class AsyncConfig:
    """Reads the config file pointed to by NAPARI_ASYNC.

    Parameters
    ----------
    data : dict
        The config settings.
    """

    def __init__(self, data: dict):
        LOGGER.info("AsyncConfig.__init__ config = ")
        LOGGER.info(json.dumps(data, indent=4, sort_keys=True))
        self.data = data
        _log_to_file(self.log_path)

    @property
    def synchronous(self) -> bool:
        """True if loads should be done synchronously."""
        return self.data.get("synchronous", True)

    @property
    def num_workers(self) -> int:
        """The number of worker threads or processes to create."""
        return self.data.get("num_workers", 6)

    @property
    def log_path(self) -> str:
        """The file path where the log file should be written."""
        return self.data.get("log_path")

    @property
    def use_processes(self) -> bool:
        """True if we should use processes instead of threads."""
        return self.data.get("use_processes", False)

    @property
    def delay_seconds(self) -> float:
        """The number of seconds to delay before initiating a load.

        The default of 100ms makes sure that we don't spam the workers with
        tons of load requests while scrolling with the the slider.

        The data from those loads would just be ignored since we'd no
        longer be on those slices when the load finished, so it would waste
        bandwidth. Also it might me no worker was available we finally do
        settle on a slice that we care about.
        """
        return self.data.get("delay_seconds", 0.1)

    @property
    def load_seconds(self) -> float:
        """Sleep for this many seconds in the worker during load.

        This is only usefull during debugging or development. Basically it
        simulates a slow internet connection or slow computation.
        """
        return self.data.get("load_seconds", 0)


def _load_config(config_path: str) -> dict:
    """Load the JSON formatted config file.

    config_path : str
        The file path of the JSON file we should load.
    """

    path = Path(config_path).expanduser()
    if not path.exists():
        # Example exception message:
        #     "Config file NAPARI_ASYNC=missing-file.json not found"
        raise FileNotFoundError(
            errno.ENOENT,
            f"Config file {ASYNC_ENV_VAR}={path} not found",
            path,
        )

    with path.open() as infile:
        return json.load(infile)


def _get_config_data() -> dict:
    """Return the user's config file data or a default config.
    """
    value = os.getenv(ASYNC_ENV_VAR)

    if value is None or value == "0":
        return DEFAULT_SYNC_CONFIG  # Async is disabled.
    elif value == "1":
        return DEFAULT_ASYNC_CONFIG  # Async is enabled with defaults.
    else:
        return _load_config(value)  # Load the user's config file.


# The global instance
async_config = AsyncConfig(_get_config_data())
