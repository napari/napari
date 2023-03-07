"""Async and Octree config file.

Async/octree has its own little JSON config file. This is temporary
until napari has a system-wide one.
"""
import logging
from typing import Optional

from napari.settings import get_settings
from napari.utils.translations import trans

LOGGER = logging.getLogger("napari.loader")
DEFAULT_OCTREE_CONFIG = {
    "loader_defaults": {
        "log_path": None,
        "force_synchronous": False,
        "num_workers": 10,
        "use_processes": False,
        "auto_sync_ms": 30,
        "delay_queue_ms": 100,
    },
    "octree": {
        "enabled": False,
        "tile_size": 256,
        "log_path": None,
        "loaders": {
            0: {"num_workers": 10, "delay_queue_ms": 100},
            2: {"num_workers": 10, "delay_queue_ms": 0},
        },
    },
}


def _get_async_config() -> Optional[dict]:
    """Get configuration implied by NAPARI_ASYNC.

    Returns
    -------
    Optional[dict]
        The async config to use or None if async not specified.
    """

    async_var = get_settings().experimental.async_

    if async_var in [True, False]:
        async_var = str(int(async_var))

    # NAPARI_ASYNC can now only be "0" or "1".
    if async_var not in [None, "0", "1"]:
        raise ValueError(
            trans._(
                'NAPARI_ASYNC can only be "0" or "1"',
                deferred=True,
            )
        )

    # If NAPARI_ASYNC is "1" use defaults but with octree disabled.
    if async_var == "1":
        async_config = DEFAULT_OCTREE_CONFIG.copy()
        async_config['octree']['enabled'] = False
        return async_config

    # NAPARI_ASYNC is not enabled.
    return None


def get_octree_config() -> dict:
    """Return the config data from the user's file or the default data.

    Returns
    -------
    dict
        The config data we should use.
    """
    # TODO the following is commented out to disallow setting octree via
    # environment variables and forcing octree to always be False

    # settings = get_settings()
    # octree_var = settings.experimental.octree

    # if octree_var in [True, False]:
    #     octree_var = str(int(octree_var))

    # # If NAPARI_OCTREE is not enabled, defer to NAPARI_ASYNC
    # if octree_var in [None, "0"]:
    #     # This will return DEFAULT_ASYNC_CONFIG or None.
    #     return _get_async_config()

    # # If NAPARI_OCTREE is "1" then use default config.
    # if octree_var == "1":
    #     return DEFAULT_OCTREE_CONFIG

    # # NAPARI_OCTREE should be a config file path
    # path = Path(octree_var).expanduser()
    # with path.open() as infile:
    #     json_config = json.load(infile)

    # # Need to set this for the preferences dialog to build.
    # settings.experimental.octree = False
    json_config = DEFAULT_OCTREE_CONFIG
    return json_config
