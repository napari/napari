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

Enabling the Octree automatically enables the
:class:`~napari.components.experimental.chunk._loader.ChunkLoader`.

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

Tiled Visuals
-------------

The visual portion of Octree rendering is implemented by three classes:
:class:`~napari._vispy.experimental.vispy_tiled_image_layer.VispyTiledImageLayer`,
:class:`~napari._vispy.experimental.vispy_tiled_image_visual.TiledImageVisual`,
and :class:`~napari._vispy.experimental.texture_atlas.TextureAtlas2D`.

The first two classes are named "tiled image" rather than "octree" because
currently they do not know that they are rendering out of an octree. We did
this to keep the visuals simpler and more general, however the approach has
some limitations, and we need need to create a subclass of
:class:`~napari._vispy.experimental.vispy_tiled_image_visual.TiledImageVisual`
which is Octree-specific at some point.

The :class:`~napari._vispy.experimental.texture_atlas.TextureAtlas2D` class
is a subclass of the basic Vispy `Texture2D` class. Its key method is
:meth:`~napari._vispy.experimental.texture_atlas.TextureAtlas2D.add_tile`
which adds a next texture to the atlas. Each texture in the atlas has
texture coordinates which denotes a tile-sized rectangle within the full
texture, and vertices which denote where that texture should be drawn in
the scene.

Future Work: Multiple Tile Sizes
--------------------------------

Today all tiles in the texture atlas have to be the same size. However the
coarsest level in multiscale datasets in the wild are often much bigger
than our tile size. Today we solve that with a method
:meth:`~napari.layers.image.experimental.octree_image.OctreeImage._create_extra_levels`
that adds levels to the multiscale data until the coarsest level fits
within a single tile.

This is not a great solution. It's potentially quite slow to add these
additional levels, since it involves downsampling.  It would be better if
we could make an exception for the highest level and allow its tile size to
be bigger than what we use in the rest of the tree. As long as it smaller
than the max texture size, which is (16384, 16384) on some hardware.

This is also probably a necessary step if we want `OctreeImage` to someday
replace `Image`. If an image is smaller than the max texture size, in at
least some cases we probably want to draw that image as a single tile.  If
`OctreeImage` is going to replace `Image` we probably want to avoid
unnecessarily tiling images that do not need to be tiled.


Octree Rendering
----------------
The interface between the visuals and the Octree is the `OctreeImage`
method
:meth:`~napari.layers.image.experimental.octree_image.OctreeImage.get_drawable_chunks`.
The method is called by
:meth:`~napari._vispy.experimental.vispy_tiled_image_layer.VispyTiledImageLayer._update_drawn_chunks`
every frame so it can update which tiles are drawn. The
:class:`~napari.layers.image.experimental.octree_image.OctreeImage` calls
the `get_intersection()` on its
:class:`~napari.layers.image.experimental._octree_slice.OctreeSlice` to get
an
:class:`~napari.layers.image.experimental.octree_intersection.OctreeIntersection`
object which contains the "ideal chunks" that should be drawn for the
current camera position.

The ideal chunks are the ones at the preferred level of detail, the level
of detail that best matches the current canvas resolution. Drawing chunks
which are more detailed that this will look fine, the graphics card will
downsample them, but it will take longer. Drawing chunks that are coarser
than the ideal level will look blurry, but it's better than drawing nothing.

The decision about what level of detail to use is made by the
:class:`~napari.layers.image.experimental._octree_loader.OctreeLoader`
class and its method
:meth::`~napari.layers.image.experimental._octree_loader.OctreeLoader.get_drawable_chunks`.
In addition to deciding what level of detail to draw for each ideal chunk,
the class initiates asynchronous loads with the
:class:`~napari.components.experimental.chunk._loader.ChunkLoader` for
chunks it wants to draw in the future.

The basic algorithm is the loader will only use chunks from a higher
resolution if they are already being drawn. It will never initiate loads on
higher resolution chunks, because it's better off loading the ideal chunks.

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
