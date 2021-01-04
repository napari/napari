.. _rendering:

Napari Async Rendering
======================

As discussed in the rendering backgrounder, in order for napari to remain
responsive we cannot access array-like data in the GUI thread. Array-like
objects such as `Dask <https://dask.org>`_ arrays can perform IO or
computations when they are accessed, and that IO or computation might take
a very long time.

Instead using the experimental
:class:`~napari.components.experimental.chunk._loader.ChunkLoader` class we
access the data in a worker thread. Meanwhile napari renders without
blocking using the data which has already been loaded.


Enabling Async Rendering
------------------------

There are two experimental rendering features which you can opt-in to using
with the following environment variables:

Set `NAPARI_ASYNC=1` to use the regular `Image` class with the experimental
`ChunkLoader` for asynchronous loading.

Set `NAPARI_OCTREE=1` to use the experimental
:class:`~napari.layers.image.experimental.octree_image.OctreeImage` class
with the experimental
:class:`~napari.components.experimental.chunk._loader.ChunkLoader` and the
experimental
:class:`~napari._vispy.experimental.tiled_image_visual.TiledImageVisual`
for tiled rendering.

NAPARI_ASYNC
------------

When the :class:`~napari.components.experimental.chunk._loader.ChunkLoader`
is enabled with the regular :class:`~napari.layers.image.image.Image` both
single-scale and multi-scale images are loaded asynchronously.

Single-scale images
^^^^^^^^^^^^^^^^^^^

Without asynchronous loading, when scrolling through slices of a single
scale image, you must wait the full duration of each slice loading before
you can advance to the next slice. If the images are load to load, this can
lead the user to feel stuck, unable to freely select a slice. It can also
lead to the spinning wheel of death if the slice takes a really long time
to load.

With asynchronous loading you can freely change slices even if the current
slice has not loaded yet. The asynchronous load that was in progress is
either aborted, or it just completes in the background and has no effect.

Multi-scale images
^^^^^^^^^^^^^^^^^^

The :class:`~napari.layers.image.image.Image` class implements multi-scale
rendering without using tiles. When the image is panned or zoom, the entire
contents of the current view of the data is loaded. This works surprisingly
well for local data, but it's quite slow for remote data because even if
you pan only a tiny bit, it gets an entire window's worth of data.

Still asynchronous rendering improves the experience and prevents napari
from seeming hung and showing the spinning wheel of death.

NAPARI_OCTREE
-------------

Setting `NAPARI_OCTREE=1` enables Octree with the default configuration. To
customize the configuration set `NAPARI_OCTREE` to be the path of a JSON
config file, such as `NAPARI_OCTREE=/tmp/octree.json`

See :data:`~napari.utils._octree.DEFAULT_OCTREE_CONFIG` for the current
config file format, for example:

.. code-block:: python
    {
        "loader_defaults": {
            "log_path": None,
            "force_synchronous": False,
            "num_workers": 10,
            "use_processes": False,
            "auto_sync_ms": 30,
            "delay_queue_ms": 100,
        },
        "octree": {
            "enabled": True,
            "tile_size": 256,
            "log_path": None,
            "loaders": {
                0: {"num_workers": 10, "delay_queue_ms": 100},
                2: {"num_workers": 10, "delay_queue_ms": 0},
            },
        },
    }





