"""AsyncConfig and OctreeConfig.

Async/octree has its own little JSON config file. This is temporary because
napari does not yet its own system-wide config file. Once that config file
exists, we can roll these settings into there.
"""
import errno
import json
import logging
import os
from collections import namedtuple
from pathlib import Path

from ....utils import config

LOGGER = logging.getLogger("napari.async")

# NAPARI_ASYNC=0 will use these settings, although currently with async
# in experimental this module will not even be imported if NAPARI_ASYNC=0.
DEFAULT_SYNC_CONFIG = {"force_synchronous": True}

# NAPARI_ASYNC=1 will use these default settings:
DEFAULT_ASYNC_CONFIG = {
    "log_path": None,
    "force_synchronous": False,
    "num_workers": 6,
    "use_processes": False,
    "auto_sync_ms": 30,
    "delay_queue_ms": 100,
}

# The async config settings.
AsyncConfig = namedtuple(
    "AsyncConfig",
    [
        "log_path",
        "force_synchronous",
        "num_workers",
        "use_processes",
        "auto_sync_ms",
        "delay_queue_ms",
        "octree",
    ],
)

# The octree config settings.
OctreeConfig = namedtuple("OctreeConfig", ["tile_size"],)

DEFAULT_OCTREE_CONFIG = {"tile_size": 256}


def _log_to_file(path: str) -> None:
    """Log "napari.async" messages to the given file.

    Parameters
    ----------
    path : str
        Log to this file path.
    """
    if path:
        fh = logging.FileHandler(path)
        LOGGER.addHandler(fh)
        LOGGER.setLevel(logging.DEBUG)


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
        # Produce a nice error message like:
        #     Config file NAPARI_ASYNC=missing-file.json not found
        raise FileNotFoundError(
            errno.ENOENT, f"Config file NAPARI_ASYNC={path} not found", path,
        )

    with path.open() as infile:
        return json.load(infile)


def _get_config_data() -> dict:
    """Return the config data from the user's file or the default data.

    Return
    ------
    dict
        The config data we should use.
    """
    value = os.getenv("NAPARI_ASYNC")

    if (value is None or value == "0") and not config.async_octree:
        # Async is disabled. Note, while async is experimental if
        # NAPARI_ASYNC and NAPARI_OCTREE are both not set, then actually
        # this code will not be run. No async related code is even
        # imported.
        #
        # However long term, using DEFAULT_SYNC_CONFIG means the
        # ChunkLoader is being used, but ChunkLoader.force_synchronous is
        # set True. Meaning every single load is synchronous.
        #
        # Once the feature is not experimental that will probably
        # be the way to turn async "off". It will still run through
        # the ChunkLoader, but the loads will be synchronous.
        return DEFAULT_SYNC_CONFIG

    # Async is enabled with defaults.
    async_defaults = DEFAULT_ASYNC_CONFIG
    async_defaults['octree'] = DEFAULT_OCTREE_CONFIG
    return async_defaults

    return _load_config(value)  # Load the user's JSON config file.


def _create_async_config(data: dict) -> AsyncConfig:
    """Creates the AsyncConfig named tuple and set up logging.

    Parameters
    ----------
    data : dict
        The config settings.

    Return
    ------
    AsyncConfig
        The config settings to use.
    """
    octree_data = data.get("octree")
    if octree_data is None:
        octree_data = DEFAULT_OCTREE_CONFIG

    octree_config = OctreeConfig(tile_size=octree_data.get("tile_size", 64))

    config = AsyncConfig(
        log_path=data.get("log_path"),
        force_synchronous=data.get("synchronous", True),
        num_workers=data.get("num_workers", 6),
        use_processes=data.get("use_processes", False),
        auto_sync_ms=data.get("auto_sync_ms", 30),
        delay_queue_ms=data.get("delay_queue_ms", 100),
        octree=octree_config,
    )

    _log_to_file(config.log_path)
    LOGGER.info("_create_async_config = ")
    LOGGER.info(json.dumps(data, indent=4, sort_keys=True))

    return config


# The global config settings instance.
async_config = _create_async_config(_get_config_data())
