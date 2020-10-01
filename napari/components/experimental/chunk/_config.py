"""AsyncConfig to configure asynchronous loading and the ChunkLoader.
"""
import errno
import json
import logging
import os
from collections import namedtuple
from pathlib import Path

LOGGER = logging.getLogger("ChunkLoader")

# Use NAPARI_ASYNC to enable or configure async.
ASYNC_ENV_VAR = "NAPARI_ASYNC"

# NAPARI_ASYNC=0 or missing config will use these settings:
DEFAULT_SYNC_CONFIG = {"synchronous": True}

# NAPARI_ASYNC=1 will use these settings:
DEFAULT_ASYNC_CONFIG = {
    "log_path": None,
    "synchronous": False,
    "num_workers": 6,
    "auto_sync_ms": 30,
    "delay_queue_ms": 100,
}

# Config settings for ChunkLoader and other async stuff.
AsyncConfig = namedtuple(
    "AsyncConfig",
    [
        "log_path",
        "synchronous",
        "num_workers",
        "auto_sync_ms",
        "delay_queue_ms",
    ],
)


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


def _load_config(config_path: str) -> dict:
    """Load the JSON formatted config file.

    config_path : str
        The file path of the JSON file we should load.
    """

    path = Path(config_path).expanduser()
    if not path.exists():
        # Example error: "Config file NAPARI_ASYNC=missing-file.json not found"
        raise FileNotFoundError(
            errno.ENOENT,
            f"Config file {ASYNC_ENV_VAR}={path} not found",
            path,
        )

    with path.open() as infile:
        return json.load(infile)


def _get_config_data() -> dict:
    """Return the config data from the user's file or the default.
    """
    value = os.getenv(ASYNC_ENV_VAR)

    if value is None or value == "0":
        return DEFAULT_SYNC_CONFIG  # Async is disabled.
    elif value == "1":
        return DEFAULT_ASYNC_CONFIG  # Async is enabled with defaults.
    else:
        return _load_config(value)  # Load the user's config file.


def _create_async_config(data: dict) -> AsyncConfig:
    """Creates the AsyncConfig object and sets up logging.

    Parameters
    ----------
    data : dict
        The config settings.
    """
    config = AsyncConfig(
        data.get("log_path"),
        data.get("synchronous", True),
        data.get("num_workers", 6),
        data.get("async_sync_ms", 30),
        data.get("delay_queue_ms", 0.1),
    )

    _log_to_file(config.log_path)
    LOGGER.info("_create_async_config = ")
    LOGGER.info(json.dumps(data, indent=4, sort_keys=True))

    return config


# The global instance
async_config = _create_async_config(_get_config_data())
