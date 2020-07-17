"""AsyncConfig to configure ChunkLoader and related stuff.
"""
import errno
import json
import logging
import os
from pathlib import Path

ASYNC_ENV_VAR = "NAPARI_ASYNC"

DEFAULT_SYNC_CONFIG = {"synchronous": True}

DEFAULT_ASYNC_CONFIG = {
    "synchronous": False,
    "num_workers": 6,
    "log_path": None,
}

LOGGER = logging.getLogger("ChunkLoader")


def _log_to_file(path):
    """Write to log file specified in the config."""
    if path is not None:
        fh = logging.FileHandler(path)
        LOGGER.addHandler(fh)
        LOGGER.setLevel(logging.INFO)


class AsyncConfig:
    """Reads the config file pointed to by NAPARI_ASYNC.
    """

    def __init__(self, data: dict):
        self.data = data
        _log_to_file(self.log_path)

    @property
    def synchronous(self):
        return self.data.get("synchronous", True)

    @property
    def num_workers(self):
        return self.data.get("num_workers")

    @property
    def log_path(self):
        return self.data.get("log_path")


def _load_config(config_path: str):
    """Load the JSON formatted config file.
    """

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            errno.ENOENT, f"Config file {ASYNC_ENV_VAR} not found", path,
        )

    with path.open() as infile:
        return json.load(infile)


def _get_config_data():
    """Load the user's config file or use a default.
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
