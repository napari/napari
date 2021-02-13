"""Napari Configuration.
"""
import os

from ._octree import get_octree_config


def _set(env_var: str) -> bool:
    """Return True if the env variable is set and non-zero.

    Returns
    -------
    bool
        True if the env var was set to a non-zero value.
    """
    return os.getenv(env_var) not in [None, "0"]


"""
Experimental Features

Async Loading
-------------
Image layers will use the ChunkLoader to load data instead of loading
the data directly. Image layers will not call np.asarray() in the GUI
thread. The ChunkLoader will call np.asarray() in a worker thread. That
means any IO or computation done as part of the load will not block the
GUI thread.

Set NAPARI_ASYNC=1 to turn on async loading with default settings.

Octree Rendering
----------------
Image layers use an octree for rendering. The octree organizes the image
into chunks/tiles. Only a subset of those chunks/tiles are loaded and
drawn at a time. Octree rendering is a work in progress.

Enabled one of two ways:

1) Set NAPARI_OCTREE=1 to enabled octree rendering with defaults.

2) Set NAPARI_OCTREE=/tmp/config.json use a config file.

See napari/utils/_octree.py for the config file format.

Shared Memory Server
--------------------
Experimental shared memory service. Only enabled if NAPARI_MON is set to
the path of a config file. See this PR for more info:
https://github.com/napari/napari/pull/1909.
"""

# Config for async/octree. If octree_config['octree']['enabled'] is False
# only async is enabled, not the octree.
octree_config = get_octree_config()

# Shorthand for async loading with or without an octree.
async_loading = octree_config is not None

# Shorthand for async with an octree.
async_octree = octree_config and octree_config['octree']['enabled']

# Shared Memory Server
monitor = _set("NAPARI_MON")

"""
Other Config Options
"""
# Added this temporarily for octree debugging. The welcome visual causes
# breakpoints to hit in image visual code. It's easier if we don't show it.
allow_welcome_visual = True
