.. _rendering:

Asynchronous Rendering
======================

As discussed in the explanations document on rendering, asynchronous
rendering is a feature that allows napari to stay usable and responsive
even when data is loading slowly. The two experimental asynchronous
rendering features can be enabled using the environment variables
``NAPARI_ASYNC`` and ``NAPARI_OCTREE``.

NAPARI_ASYNC
------------

Running napari with ``NAPARI_ASYNC=1`` enables asynchronous rendering using
the existing :class:`~napari.layers.image.image.Image` class. The
:class:`~napari.layers.image.image.Image` class will no longer call
``np.asarray()`` in the GUI thread. This way if ``np.asarray()`` blocks on
IO or a computation, the GUI thread will not block and the framerate will
not slow down.

To avoid blocking the GUI thread the
:class:`~napari.layers.image.image.Image` classes loads chunks using the
:class:`~napari.components.experimental.chunk._loader.ChunkLoader`. It will
call ``np.asarray()`` in a worker thread or worker process. When the worker
thread finishes it will call
:meth:`~napari.layers.image.image.Image.on_chunk_loaded` with the loaded
data. The next frame :class:`~napari.layers.image.image.Image` can display
the new data.

Time-series Data
^^^^^^^^^^^^^^^^

Without ``NAPARI_ASYNC`` napari will block when switching slices. Napari
will hang until the new slice has loaded. If the slice loads slowly enough
you might see the "spinning wheel of death" on a Mac indicating the process
is hung.

Asynchronous rendering allows the user to interrupt the loading of a slice
at any time. The user can freely move the slice slider. This is especially
nice for remote or slow-loading data.

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
cannot interrupt the load to further adjust the camera.

With ``NAPARI_ASYNC`` performance is the same, however you can interrupt
the load by moving the camera at any time. This is a nice improvement, but
working with slow-loading data is still awkward. Most large image viewers
improve on this experience with chunks or tiles. With chunks or tiles when
the image is panned the existing tiles are just translated. Then the viewer
only needs to fetch tiles which newly slid onto the screen. This style of
rendering what our ``NAPARI_OCTREE`` flag enables.

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

See `Octree Configuration File`_ for Octree configuration options.

Octree Visuals
^^^^^^^^^^^^^^

The visual portion of Octree rendering is implemented by three classes:
:class:`~napari._vispy.experimental.vispy_tiled_image_layer.VispyTiledImageLayer`,
:class:`~napari._vispy.experimental.vispy_tiled_image_visual.TiledImageVisual`,
and :class:`~napari._vispy.experimental.texture_atlas.TextureAtlas2D`.

The first two classes are named "tiled image" rather than "octree" because
currently they do not know that they are rendering out of an octree. We did
this intentionally to keep the visuals simpler and more general. However,
the approach has some limitations, and we might later need to create a
subclass of
:class:`~napari._vispy.experimental.vispy_tiled_image_visual.TiledImageVisual`
which is Octree-specific. To get all the octree rendering behaviors we
want.

The :class:`~napari._vispy.experimental.texture_atlas.TextureAtlas2D` class
is a subclass of the generic Vispy ``Texture2D`` class. Like ``Texture2D``
the :class:`~napari._vispy.experimental.texture_atlas.TextureAtlas2D` class
owns one texture. However
:class:`~napari._vispy.experimental.texture_atlas.TextureAtlas2D` uses this
one texture as an "atlas" which can hold multiple tiles.

For example, by default
:class:`~napari._vispy.experimental.texture_atlas.TextureAtlas2D` uses a
(4096, 4096) texture that stores 256 different (256, 256) pixel tiles.
Adding or remove a single tile from the full atlas texture is very fast.
Under the hood adding one tile calls ``glTexSubImage2D()`` which only
updates the data in that one (256, 256) portion of the full texture.

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
:meth:`~napari.layers.image.experimental._octree_slice.OctreeSlice.get_intersection`
method on its
:class:`~napari.layers.image.experimental._octree_slice.OctreeSlice` to get
an
:class:`~napari.layers.image.experimental.octree_intersection.OctreeIntersection`
object which contains the "ideal chunks" that should be drawn for the
current camera position.

The ideal chunks are the chunks at the preferred level of detail, the level
of detail that best matches the current canvas resolution. Drawing chunks
which are more detailed that this will look fine, the graphics card will
downsample them, but it's not efficient to use higher resolution chunks
than are needed. Meanwhile drawing chunks that are coarser than the ideal
level will look blurry, but it's much better than drawing nothing.

The decision about what level of detail to use is made by the
:class:`~napari.layers.image.experimental._octree_loader.OctreeLoader`
class and its method
:meth:`~napari.layers.image.experimental._octree_loader.OctreeLoader.get_drawable_chunks`.
There are many different approaches one could take here as far as what to
draw when, today we are doing something reasonable but it could potentially
be improved. In addition to deciding what level of detail to draw for each
ideal chunk, the class initiates asynchronous loads with the
:class:`~napari.components.experimental.chunk._loader.ChunkLoader` for
chunks it wants to draw in the future.

The loader will only use chunks from a higher resolution if they are
already being drawn. For example when zooming out. However, it will never
initiate loads on higher resolution chunks, since it's better off loading
and drawing the ideal chunks.

The loader will load lower resolution chunks in some cases. Although this
can slightly delay when the ideal chunks are loaded, it's a very quick way
to get reasonable looking "coverage" of the area of interest. Often data
from one or two levels up isn't even that noticeably degraded. This table
shows how many ideal chunks are "covered" a chunk at a higher level:

==================  ======
Levels Above Ideal  Coverage
------------------  ------
1                   4
2                   16
3                   64
==================  ======

Octree Configuration File
^^^^^^^^^^^^^^^^^^^^^^^^^

Setting ``NAPARI_OCTREE=1`` enables Octree rendering with the default
configuration. To customize the configuration set ``NAPARI_OCTREE`` to be
the path of a JSON config file, such as ``NAPARI_OCTREE=/tmp/octree.json``.

See :data:`~napari.utils._octree.DEFAULT_OCTREE_CONFIG` for the current
config file format:

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
:class:`~napari.components.experimental.chunk._loader.ChunkLoader`.

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

The ``octree`` key contains these settings:

+-----------------------+-----------------------------------------------------------+
| Setting               | Description                                               |
+=======================+===========================================================+
| ``enabled``           | If ``false` then use the old `Image` class.               |
+-----------------------+-----------------------------------------------------------+
| ``tile_size``         | Size of render tiles to use for rending.                  |
+-----------------------+-----------------------------------------------------------+
| ``log_path``          | Octree specific log file for debugging.                   |
+-----------------------+-----------------------------------------------------------+
| ``loaders``           | Optional custom loaders, see below.                       |
+-----------------------+-----------------------------------------------------------+

The ``loaders`` key lets you define and configure multiple
:class:`~napari.components.experimental.chunk._pool.LoaderPool` pools. The
key of each loader is the levels relative to the ideal level. In the above
example configuration we define two loaders. The first with key "0" is for
loading chunks at the ideal level or one above. While the second with key
"2" will load chunks two above the ideal level or higher.

Each loader uses the ``loader_defaults`` but you can override the
``num_workers``, ``auto_sync_ms`` and ``delay_queue_ms`` values in
each loader defined in ``loaders``.

Multiple Loaders
^^^^^^^^^^^^^^^^

We allow multiple loaders to improve loading performance. There are a lot
of different strategies one could use when loading chunks. For example,
we tend to load chunks at a higher level prior to loading the chunks
at the ideal level. This gets "coverage" on the screen quickly, and then
the data can be refined by loading the ideal chunks.

One consideration is during rapid movement of the camera it's easy to clog
up the loader pool with workers loading chunks that have already moved out
of view. The
:class:`~napari.components.experimental.chunk._delay_queue.DelayQueue` was
created to help with this problem.

While we can't cancel a load if a worker as started working on it, we can
trivially cancel loads that are still in our delay queue. If the chunk goes
out of view, we cancel the load. If the user pauses for a bit, we initiate
the loads.

With multiple loaders we can delay the ideal chunks, but we can configure
zero delay for the higher levels. A single chunk from two levels up will
cover 16 ideal chunks. So immediately loading them is a good way to get
data on the screen quickly. When the camera stops moving the
:class:`~napari.components.experimental.chunk._pool.LoaderPool` for the
ideal layer will often be empty. So all of those workers can immediately
start loading the ideal chunks.

The ability to have multiple loaders was only recently added. We still need
to experiment to figure out the best configuration. And figure out how that
configuration needs to vary based on latency of the data or other
considerations.

Future Work: Extending TextureAtlas2D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We could improve our
:class:`~napari._vispy.experimental.texture_atlas.TextureAtlas2D` class in
a number of ways:

1. Support setting the atlas's full texture size on the fly.
2. Support setting the atlas's tile size on the fly.
3. Support a mix of tiles sizes in one atlas.
4. Allow an atlas to have more than one backing texture.

One reason to consider these changes is so we could support "large tiles"
in certain cases. Often the coarsest level of multi-scale data "in the
wild" is much bigger than one of our (256, 256) tiles. Today we solve that
by creating additional Octree levels, downsampling the data until the
coarsest level fits within a single tile.

If we could support multiple tiles sizes and multiple backing textures, we
could potentially have "interior tiles" which were small, but then allow
large root tiles. Graphics card handle pretty big textures. A layer that's
(100000, 100000) obviously needs to be broken into tiles But a layers that's
(4096, 4096) is really not that big. That could be a single tile.

Long term it would be nice if we did not have to support two images
classes: :class:`~napari.layers.image.image.Image` and
:class:`~napari.layers.image.experimental.octree_image.OctreeImage`.
Maintain two code paths and two sets of visuals will become tiresome and
lead to discrepancies in how things are rendered

Instead it would be nice if
:class:`~napari.layers.image.experimental.octree_image.OctreeImage` become
the only image class. For that to happen though, we need to render small
images just as efficinetly as we do today, which probably means not
breaking them into tiles. To do this our atlas textures need to support
tiles of various sizes.

Future Work: Level Zero Only Octrees
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In issue `#1300 <https://github.com/napari/napari/issues/1300>`_ it takes
1500ms to switch slices in a (16384, 16384) image that is entirely in RAM.
The delay is not from loading into RAM, it's already in RAM, the delay is
from transferring all that data to VRAM.

The image is not a multi-scale image. Generally we've found downsampling to
create multi-scale image layers is slow. So the question is how can we draw
this large texture without hanging? One idea is we could create an Octree
that only has a level zero and no downsampled levels.

This is an option because chopping up a ``numpy`` array into tiles is very
fast, because no memory is moved. It's really just creating a bunch of
"views" into the single existing array. So creating a level zero Octree
should be very fast. For there we can use our existing Octree code and our
existing
:class:`~napari._vispy.experimental.vispy_tiled_image_visual.TiledImageVisual`
to transfer over one tile at a time without hurting the frame rate.

The insight here is our Octree code is really implemented two things, one
is an Octree but two is a tiled or chunked image, which is just a grid of
tiles. It's TBD how this would look to the user. But instead of a 1500ms
hang they'd probably see the individuals tiles peppering into the scene.
And they could interrupt this process by switching slices at any point.
