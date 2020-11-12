"""Napari Configuration.
"""
import os


def _set(env_var: str) -> bool:
    """Return True if the env variable is set and non-zero.

    Return
    ------
    bool
        True if the env var was set to a non-zero value.
    """
    return os.getenv(env_var, "0") != "0"


# Added this temporarily for octree debugging. The welcome visual causes
# breakpoints to hit in image visual code. It's easier if we don't show it.
allow_welcome_visual = True


"""
Experimental Image Configuration.

Async loading and octree image rendering are two separate but related
experimental features.

Async Loading
-------------
Image layers will use the ChunkLoader to load data instead of loading
the data directly. Image layers will not call np.asarray() in the GUI
thread. The ChunkLoader will call np.asarray() in a worker thread. That
means any IO or computation done as part of the load will not block the
GUI thread.

Set NAPARI_ASYNC=1 to turn on async loading with default settings.

Set NAPARI_ASYNC=/tmp/async.json to enable async loading with
a configuration file. See config settings here:
    napari.components.experimental.chunk._config.py

Octree Rendering
----------------
Image layers use an octree for rendering. The octree organizes the image
into chunks/tiles. Only a subset of those chunks/tiles are loaded and
drawn at a time. Octree rendering is very WIP and not ready yet.

Set NAPARI_OCTREE=1 to enable experimental octree visuals. This
will also turn on async loading, however to configure async
loading you can set NAPARI_ASYNC to a config file path as above.

Future
------
When we're done with octree development we want to return to one single
image class, with one single type of visual. All of this config complexity
is temporary.
"""

async_octree = _set("NAPARI_OCTREE")
async_loading = _set("NAPARI_ASYNC") or async_octree

"""
Image Layer Creation

With octree enabled the QtTestImage widget allows creating 3 types of image
layers/visuals. This lets us compare each implemention.
"""

# Image layer, the original one.
CREATE_IMAGE_NORMAL = 1

# OctreeImage layer with the VispyCompoundImageLayer visual.
# This visual uses a separate ImageVisual for each tile.
CREATE_IMAGE_COMPOUND = 2

# OctreeImage layer with the VispyTiledImageLayer visual.
# This is a single visual that can draw many tiles.
CREATE_IMAGE_TILED = 3

# The default type of image the QtTestImage widget will create. If the user
# selects another type of image, the widget will change this config value.
create_image_type = CREATE_IMAGE_TILED


def create_octree_image() -> bool:
    """Return True if we should create an OctreeImage layer.

    Only create an OctreeImage layer if octree is enabled AND we are
    creating an octree-based image type.

    Return
    ------
    bool
        True if we should create an OctreeImage layer.
    """
    return async_octree and (create_image_type != CREATE_IMAGE_NORMAL)
