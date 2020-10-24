"""Configuration.
"""
import os


def _set(var: str) -> bool:
    """Return True if the environment variable is set is not 0."""
    return os.getenv(var, "0") != "0"


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
# Image layers construct and render out of an octree. This is a WIP is
# not really functional yet. Octree rendering implies that async loading
# is enabled.
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
