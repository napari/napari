.. _rendering:

Asynchronous Rendering
======================

As discussed in the explanations document on rendering, asynchronous
rendering is a feature that allows napari to stay usable and responsive
even when data is loading very slowly. There are two experimental features
in napari that enable asynchronous rendering. The two features can be
enabled using the environment variables ``NAPARI_ASYNC`` and
``NAPARI_OCTREE``.

NAPARI_ASYNC
------------

Running napari with ``NAPARI_ASYNC=1`` enables asynchronous rendering using
the existing :class:`~napari.layers.image.image.Image` class. With
asynchronous rendering enabled the
:class:`~napari.layers.image.image.Image` class will no longer call
``np.asarray()`` in the GUI thread. We don't want to call ``np.asarray()``
in the GUI thread because when using `Dask <https://dask.org>`_  array, or
a similar array-like object, the array access can result in disk or network
IO or in computations that block the GUI thread and hurt the framerate.

Instead of calling ``np.asarray()`` in the GUI thread, when
``NAPARI_ASYNC`` is set :class:`~napari.layers.image.image.Image` will use
the :class:`~napari.components.experimental.chunk._loader.ChunkLoader`. The
:class:`~napari.components.experimental.chunk._loader.ChunkLoader` will
call ``np.asarray()`` in a worker thread or worker process. If that results
in IO or computation only the worker will block. While the worker is
blocked, rendering can continue in the GUI thread and napari will remain
responsive and usable. When the worker thread finishes it will call
:meth:`~napari.layers.image.image.Image.on_chunk_loaded` with the loaded
data. The next frame :class:`~napari.layers.image.image.Image` can display
the new data.

Time-series Data
^^^^^^^^^^^^^^^^

Without ``NAPARI_ASYNC`` napari will block when switching slices. Napari
will be unusable until the new slices has loaded. The slice loads slowly
enough you might see the "spinning wheel of death" on a Mac indicating the
process is hung.

Asynchronous rendering allows the user to interrupt the loading of a slice
at any time by advancing to the next slice, using the slice slider or
other means. This is a nice usability improvement especially with remote or
slow-loading data.

Multi-scale Images
^^^^^^^^^^^^^^^^^^

With today's :class:`~napari.layers.image.image.Image` class there are no
tiles or chunks. Instead, whenever the camera is panned or zoomed, even a
tiny bit, napari fetches all the data needed to draw the entire current
canvas.

This actually works amazingly well with local data. Fetching the whole
canvas of data each time can be quite fast. With remote or other high
latency data, however, this method can be very slow. Even if you pan only a
tiny amount, napari has to fetch the whole canvas worth of data, and you
cannot interrupt the load the move the camera.

With ``NAPARI_ASYNC`` performance is the same, however you can interrupt
the load by moving the camera. This is a nice improvement. But working with
slow-loading data is still awkward. Most large image viewers improve on
this experience using tiles. With tiles when the image is panned the
existing tiles are just translated. Then the viewer only needs to fetch
tiles which newly slid onto the screen. This tiled rendering is exactly
what is implemented with the ``NAPARI_OCTREE`` feature.

NAPARI_OCTREE
-------------
Set ``NAPARI_OCTREE=1`` to use the experimental
:class:`~napari.layers.image.experimental.octree_image.OctreeImage` class
instead of the normal :class:`~napari.layers.image.image.Image` class. The
new :class:`~napari.layers.image.experimental.octree_image.OctreeImage`
class will use the same
:class:`~napari.components.experimental.chunk._loader.ChunkLoader` that
``NAPARI_ASYNC`` enables. In addition, ``NAPARI_OCTREE`` will use the new
:class:`~napari._vispy.experimental.tiled_image_visual.TiledImageVisual`
instead of the Vispy ``ImageVisual`` class that napari's
:class:`~napari.layers.image.image.Image` class uses.

See :ref:`Octree Configuration File` for Octree configuration options.

Octree Visuals
^^^^^^^^^^^^^^

The visual portion of Octree rendering is implemented by three classes:
:class:`~napari._vispy.experimental.vispy_tiled_image_layer.VispyTiledImageLayer`,
:class:`~napari._vispy.experimental.vispy_tiled_image_visual.TiledImageVisual`,
and :class:`~napari._vispy.experimental.texture_atlas.TextureAtlas2D`.

The first two classes are named "tiled image" rather than "octree" because
currently they do not know that they are rendering out of an octree. We did
this intentionally to keep the visuals simpler and more general. However,
the approach has some limitations, and we might later need need to create a
subclass of
:class:`~napari._vispy.experimental.vispy_tiled_image_visual.TiledImageVisual`
which is Octree-specific to get all the octree rendering behaviors we want.

The :class:`~napari._vispy.experimental.texture_atlas.TextureAtlas2D` class
is a subclass of the basic Vispy ``Texture2D`` class. Like ``Texture2D``
the :class:`~napari._vispy.experimental.texture_atlas.TextureAtlas2D` class
uses one texture. However
:class:`~napari._vispy.experimental.texture_atlas.TextureAtlas2D` uses this
one texture as an "atlas" which can hold multiple tiles.

For example, by default
:class:`~napari._vispy.experimental.texture_atlas.TextureAtlas2D` uses a
(4096, 4096) texture that stores 256 different (256, 256) pixel tiles.
Adding or remove a single tile from the full atlas texture is very fast.
Under the hood adding one tile calls ``glTexSubImage2D()`` which only
updates the data in that one (256, 256) region of the full texture.

Aside from the data transfer cost,
:class:`~napari._vispy.experimental.texture_atlas.TextureAtlas2D` is also
fast because we do not have to modify the scene graph or rebuild any
shaders. In an early version of tiled rendering we created a new
``ImageVisual`` for every tile. This resulted in scene graph changes and
shader rebuilds. At the time the scene graph changes were causing crashes
with `PyQt5`, but the atlas approach is better for multiple reasons, so
even if that crash were fixed the atlas is a better solution.


Octree Rendering
^^^^^^^^^^^^^^^^

The interface between the visuals and the Octree is the ``OctreeImage``
method
:meth:`~napari.layers.image.experimental.octree_image.OctreeImage.get_drawable_chunks`.
The method is called by ``VispyTiledImageLayer`` method
:meth:`~napari._vispy.experimental.vispy_tiled_image_layer.VispyTiledImageLayer._update_drawn_chunks`
every frame so it can update which tiles are drawn.
:class:`~napari.layers.image.experimental.octree_image.OctreeImage` calls
the
:method:`~napari.layers.image.experimental._octree_slice.OctreeSlice.get_intersection`
method on its
:class:`~napari.layers.image.experimental._octree_slice.OctreeSlice` to get
an
:class:`~napari.layers.image.experimental.octree_intersection.OctreeIntersection`
object which contains the "ideal chunks" that should be drawn for the
current camera position.

The ideal chunks are the chunks at the preferred level of detail, the level
of detail that best matches the current canvas resolution. Drawing chunks
which are more detailed that this will look fine, the graphics card will
downsample them, but it is creating unnecessary work. Drawing chunks that
are coarser than the ideal level will look blurry, but it's much better
than drawing nothing.

The decision about what level of detail to use is made by the
:class:`~napari.layers.image.experimental._octree_loader.OctreeLoader`
class and its method
:meth:`~napari.layers.image.experimental._octree_loader.OctreeLoader.get_drawable_chunks`.
In addition to deciding what level of detail to draw for each ideal chunk,
the class initiates asynchronous loads with the
:class:`~napari.components.experimental.chunk._loader.ChunkLoader` for
chunks it wants to draw in the future.

The loader will only use chunks from a higher resolution if they are
already being drawn. However, it will never initiate loads on higher
resolution chunks, since it's better off loading and drawing the ideal
chunks.

The loader will load lower resolution chunks in some cases. Although this
can slightly delay when the ideal chunks are loaded, it's a very quick way
to get reasonable looking "coverage" of the area of interest. Often data
from one or two levels up is noticeable that degraded. This table shows how
many ideal chunks are "covered" a chunk at a higher level:

==================  ======
Levels Above Ideal  Coverage
------------------  ------
1                   4
2                   16
3                   64
==================  ======

Octree Configuration File
^^^^^^^^^^^^^^^^^^^^^^^^^

Setting `NAPARI_OCTREE=1` enables Octree with the default configuration. To
customize the configuration set `NAPARI_OCTREE` to be the path of a JSON
config file, such as `NAPARI_OCTREE=/tmp/octree.json`

See :data:`~napari.utils._octree.DEFAULT_OCTREE_CONFIG` for the current
config file format. Currently it's:

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

The ``loader_defaults`` key contains settings that will be used by the
:class:`~napari.components.experimental.chunk._loader.ChunkLoader`:

+-----------------------+-----------------------------------------------------------+
| Setting               | Description                                               |
+=======================+===========================================================+
| ``log_path``          | Write ChunkLoader log file to this path. For debugging.   |
+-----------------------+-----------------------------------------------------------+
| ``force_synchronous`` | If ``true`` the ``ChunkLoader`` always load synchronously.|
+-----------------------+-----------------------------------------------------------+
| ``num_workers``       | The number of worker threads or processes.                |
+-----------------------+-----------------------------------------------------------+
| ``use_processes``     | If ``true`` use worker processes instead of threads.      |
+-----------------------+-----------------------------------------------------------+
| ``auto_async_ms``     | Switch to synchronous if loads faster than this.          |
+-----------------------+-----------------------------------------------------------+
| ``delay_queue_ms``    | Delay loads by this much time.                            |
+-----------------------+-----------------------------------------------------------+
| ``num_workers``       | The number of worker threads or processes.                |
+-----------------------+-----------------------------------------------------------+

The ``num_workers``, ``auto_sync_ms`` and ``delay_queue_ms`` values in
``loader_defaults`` can be overridden for a specific pool under the
``octree->loaders`` setting.

Future Work: Extending TextureAtlas2D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We could improve our
:class:`~napari._vispy.experimental.texture_atlas.TextureAtlas2D` class in
a number of ways:

1. Support setting the atlas texture size on the fly.
2. Support setting the tile size on the fly.
3. Support a mix of tiles sizes in one atlas.
4. Support multiple atlas textures in a single atlas.

This would allow us to use "very large tiles" in some cases. Often the
coarsest level of multi-scale data "in the wild" is much bigger than one of
our (256, 256) tiles. Today we solve that by creating additional Octree
levels, downsampling the data until the coarsest level fits within a single
tile.

A better solution might be to use "small tiles" for the interior data, but
allow a pretty big tile as root octree level. For example we might be using
(256, 256) pixel tiles, but the root level might be (2500, 2500) and we decide
to leave that as a single tile.

Long term it would be nice if
:class:`~napari.layers.image.experimental.octree_image.OctreeImage` were
the only image class. So we did not have to support two very different
paths in the code. Two types of layers, two types of visuals, etc. However
it's probably unwise to chop up modest sizes images, like (4096, 4096),
into small tiles. When the graphics card can handle (4096, 4096) perfectly
fine.

With a flexible
:class:`~napari._vispy.experimental.texture_atlas.TextureAtlas2D` we should
choose the optimal tile size for every situation. So we'd use the
:class:`~napari.layers.image.experimental.octree_image.OctreeImage` code in
all cases. But in some cases the "octree" would be just a single (4096,
4096) texture.

Future Work: Level Zero Only Octrees
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In issue `#1300 <https://github.com/napari/napari/issues/1300>`_ it takes
1500ms to switch slices in a (16384, 16384) image that entirely in RAM. The
image is not a multi-scale image. Generally we've found downsampling to
create multi-scale image layers is slow. On thing that might were for this
case is to create an Octree that only has a level zero.

Chopping up a ``numpy`` array into tiles is very fast, because no memory is
moved. It's really just creating a bunch of "views" into the single array.
So creating a level zero Octree should be very fast. For there we can use
our existing Octree code and our existing
:class:`~napari._vispy.experimental.vispy_tiled_image_visual.TiledImageVisual`
to transfer over one tile at a time without hurting the frame rate.

It's TBD exactly how we'd display this for the user. But instead of a
1500ms hang the users would see the tiles appearing very quickly one at a
time, and they would be free to interrupt and change slices at anytime.
