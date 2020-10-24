"""Configuration.
"""
import os


def _set(env_var: str) -> bool:
    """Return True if the env variable set and non-zero.

    Return
    ------
    bool
        True if the env var was set to a non-zero value.
    """
    return os.getenv(env_var, "0") != "0"


#
# Async Experimental Code
#
# There are two separate async related experimental features:
#
# A) Async Loading
#
# Image layers will use the ChunkLoader to load data instead of loading it
# directly. Image layers will not call np.asarray() in the GUI thread. The
# ChunkLoader will call np.asarray() in a worker thread. That means any IO
# or computation done as part of the load will not block the GUI thread.
#
# B) Octree Rendering
#
# Image layers construct and render out of an octree. This is a WIP and is
# not useful yet. Octree rendering implies that async loading is enabled.
#
# Two options for async loading without the octree:
#
# 1) NAPARI_ASYNC=1
#    Async loading with default config.
#
# 2) NAPARI_ASYNC=config.json
#    Async loading with a config file.
#
# Two options for octree rendering:
#
# 1) NAPARI_OCTREE=1
#    Octree rendering plus async loading with default config.
#
# 2) NAPARI_OCTREE=1 and NAPARI_ASYNC=config.json
#    Octree rendering plus sync loading with a config file.
#
async_octree = _set("NAPARI_OCTREE")
async_loading = _set("NAPARI_ASYNC") or async_octree
