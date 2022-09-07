(rendering)=

# Asynchronous rendering

As discussed in the explanations document on rendering, asynchronous
rendering is a feature that allows napari to stay usable and responsive
even when data is loading slowly. There are two experimental asynchronous
rendering features, they can be enabled using the environment variables
`NAPARI_ASYNC` and `NAPARI_OCTREE`.

## NAPARI_ASYNC

Running napari with `NAPARI_ASYNC=1` enables asynchronous rendering using
the existing {class}`~napari.layers.Image` class. The
{class}`~napari.layers.Image` class will no longer call
`np.asarray()` in the GUI thread. We do this so that if `np.asarray()`
blocks on IO or a computation, the GUI thread will not block and the
framerate will not suffer.

To avoid blocking the GUI thread the
{class}`~napari.layers.Image` class will load chunks using the
new {class}`~napari.components.experimental.chunk._loader.ChunkLoader`
class. The
{class}`~napari.components.experimental.chunk._loader.ChunkLoader` will
call `np.asarray()` in a worker thread. When the worker thread finishes
it will call {meth}`~napari.layers.Image.on_chunk_loaded` with
the loaded data. The next frame {class}`~napari.layers.Image`
can display the new data.

### Time-series data

Without `NAPARI_ASYNC` napari will block when switching slices. Napari
will hang until the new slice has loaded. If the slice loads slowly enough
you might see the "spinning wheel of death" on a Mac indicating the process
is hung.

Asynchronous rendering allows the user to interrupt the loading of a slice
at any time. The user can freely move the slice slider. This is especially
nice for remote or slow-loading data.

### Multi-scale images

With today's {class}`~napari.layers.Image` class there are no
tiles or chunks. Instead, whenever the camera is panned or zoomed napari
fetches all the data needed to draw the entire current canvas. This
actually works amazingly well with local data. Fetching the whole canvas of
data each time can be quite fast.

With remote or other high latency data, however, this method can be very
slow. Even if you pan only a tiny amount, napari has to fetch the whole
canvas worth of data, and you cannot interrupt the load to further adjust
the camera.

With `NAPARI_ASYNC` overall performance is the same, but the advantage is
you can interrupt the load by moving the camera at any time. This is a nice
improvement, but working with slow-loading data is still slow. Most large
image viewers improve on this experience with chunks or tiles. With chunks
or tiles when the image is panned the existing tiles are translated and
re-used. Then the viewer only needs to fetch tiles which newly slid onto
the screen. This style of rendering what our `NAPARI_OCTREE` flag
enables.

## NAPARI_OCTREE

Set `NAPARI_OCTREE=1` to use the experimental
{class}`~napari.layers.image.experimental.octree_image.OctreeImage` class
instead of the normal {class}`~napari.layers.Image` class. The
new {class}`~napari.layers.image.experimental.octree_image.OctreeImage`
class will use the same
{class}`~napari.components.experimental.chunk._loader.ChunkLoader` that
`NAPARI_ASYNC` enables. In addition, `NAPARI_OCTREE` will use the new
{class}`~napari._vispy.experimental.tiled_image_visual.TiledImageVisual`
instead of the Vispy `ImageVisual` class that napari's
{class}`~napari.layers.Image` class uses.

```{note}
The current `OCTREE` implementation only fully supports a single 2D image and
may not function with 3D images or multiple images. Improving support
for 3D and multiple images is part of future work on the `OCTREE`.
```

See {ref}`octree-config` for Octree configuration options.

### Octree visuals

The visual portion of Octree rendering is implemented by three classes:
{class}`~napari._vispy.experimental.vispy_tiled_image_layer.VispyTiledImageLayer`,
{class}`~napari._vispy.experimental.vispy_tiled_image_visual.TiledImageVisual`,
and {class}`~napari._vispy.experimental.texture_atlas.TextureAtlas2D`.

The first two classes are named "tiled image" rather than "octree" because
currently they do not know that they are rendering out of an octree. We did
this intentionally to keep the visuals simpler and more general. However,
the approach has some limitations, and we might later need to create a
subclass of
{class}`~napari._vispy.experimental.vispy_tiled_image_visual.TiledImageVisual`
which is Octree-specific, see {ref}`future-work-atlas-2D`.

The {class}`~napari._vispy.experimental.texture_atlas.TextureAtlas2D` class
is a subclass of the generic Vispy ``Texture2D`` class. Like ``Texture2D``
the {class}`~napari._vispy.experimental.texture_atlas.TextureAtlas2D` class
owns one texture. However
{class}`~napari._vispy.experimental.texture_atlas.TextureAtlas2D` uses this
one texture as an "atlas" which can hold multiple tiles.

For example, by default
{class}`~napari._vispy.experimental.texture_atlas.TextureAtlas2D` uses a
(4096, 4096) texture that stores 256 different (256, 256) pixel tiles.
Adding or remove a single tile from the full atlas texture is very fast.
Under the hood adding one tile calls `glTexSubImage2D()` which only
updates the data in that specific (256, 256) portion of the full texture.

Aside from the data transfer cost,
{class}`~napari._vispy.experimental.texture_atlas.TextureAtlas2D` is also
fast because we do not have to modify the scene graph or rebuild any
shaders when a tile is added or removed. In an early version of tiled
rendering we created a new `ImageVisual` for every tile. This resulted in
scene graph changes and shader rebuilds. At the time the scene graph
changes were causing crashes with `PyQt5`, but the atlas approach is better
for multiple reasons, so even if that crash were fixed the atlas is a
better solution.

### Octree rendering

The interface between the visuals and the Octree is the
{class}`~napari.layers.image.experimental.octree_image.OctreeImage` method
{meth}`~napari.layers.image.experimental.octree_image.OctreeImage.get_drawable_chunks`.
The method is called by the
{class}`~napari._vispy.experimental.vispy_tiled_image_layer.VispyTiledImageLayer`
method
{meth}`~napari._vispy.experimental.vispy_tiled_image_layer.VispyTiledImageLayer._update_drawn_chunks`
every frame so it can update which tiles are drawn.
{class}`~napari.layers.image.experimental.octree_image.OctreeImage` calls
the
{meth}`~napari.layers.image.experimental._octree_slice.OctreeSlice.get_intersection`
method on its
{class}`~napari.layers.image.experimental._octree_slice.OctreeSlice` to get
an
{class}`~napari.layers.image.experimental.octree_intersection.OctreeIntersection`
object which contains the "ideal chunks" that should be drawn for the
current camera position.

The ideal chunks are the chunks at the preferred level of detail, the level
of detail that best matches the current canvas resolution. Drawing chunks
which are more detailed that this will look fine, the graphics card will
downsample them to the screen resolution, but it's not efficient to use
higher resolution chunks than are needed. Meanwhile drawing chunks that are
coarser than the ideal level will look blurry, but it's much better than
drawing nothing.

The decision about what level of detail to use is made by the
{class}`~napari.layers.image.experimental._octree_loader.OctreeLoader`
class and its method
{meth}`~napari.layers.image.experimental._octree_loader.OctreeLoader.get_drawable_chunks`.
There are many different approaches one could take here as far as what to
draw when. Today we are doing something reasonable but it could potentially
be improved. In addition to deciding what level of detail to draw for each
ideal chunk, the class initiates asynchronous loads with the
{class}`~napari.components.experimental.chunk._loader.ChunkLoader` for
chunks it wants to draw in the future.

The loader will only use chunks from a higher resolution if they are
already being drawn. For example when zooming out. However, it will never
initiate loads on higher resolution chunks, since it's better off loading
and drawing the ideal chunks.

The loader will load lower resolution chunks in some cases. Although this
can slightly delay when the ideal chunks are loaded, it's a very quick way
to get reasonable looking "coverage" of the area of interest. Often data
from one or two levels up isn't even that noticeably degraded. This table
shows how many ideal chunks are "covered" by a chunk at a higher level:

| Levels Above Ideal | Coverage |
| -----------------: | -------: |
| 1                  | 4        |
| 2                  | 16       |
| 3                  | 64       |

Although data 3 levels above will be quite blurry, it's pretty amazing you
can load one chunk and it will cover 64 ideal chunks. This is the heart of
the power of Octrees, Quadtrees or multiscale images.

(octree-config)=
### Octree configuration file

Setting `NAPARI_OCTREE=1` enables Octree rendering with the default
configuration. To customize the configuration set `NAPARI_OCTREE` to be
the path of a JSON config file, such as `NAPARI_OCTREE=/tmp/octree.json`.

See {data}`~napari.utils._octree.DEFAULT_OCTREE_CONFIG` for the current
config file format:

```python
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
```

The `loader_defaults` key contains settings that will be used by the
{class}`~napari.components.experimental.chunk._loader.ChunkLoader`.

| Setting               | Description                                                |
| :-------------------- | :--------------------------------------------------------- |
| `log_path`            | Write `ChunkLoader` log file to this path. For debugging.  |
| `force_synchronous`   | If `true` the `ChunkLoader` loads synchronously.           |
| `num_workers`         | The number of worker threads or processes.                 |
| `use_processes`       | If `true` use worker processes instead of threads.         |
| `auto_async_ms`       | Switch to synchronous if loads are faster than this.       |
| `delay_queue_ms`      | Delay loads by this much.                                  |
| `num_workers`         | The number of worker threads or processes.                 |

The `octree` key contains these settings:

| Setting               | Description                                                |
| :-------------------- | :--------------------------------------------------------- |
| `enabled`             | If `false` then use the old `Image` class.                 |
| `tile_size`           | Size of render tiles to use for rending.                   |
| `log_path`            | Octree specific log file for debugging.                    |
| `loaders`             | Optional custom loaders, see below.                        |

The `loaders` key lets you define and configure multiple
{class}`~napari.components.experimental.chunk._pool.LoaderPool` pools. The
key of each loader is the level relative to the ideal level. In the above
example configuration we define two loaders. The first with key `0` is for
loading chunks at the ideal level or one above. While the second with key
`2` will load chunks two above the ideal level or higher.

Each loader uses the `loader_defaults` but you can override the
`num_workers`, `auto_sync_ms` and `delay_queue_ms` values in
each loader defined in `loaders`.

### Multiple loaders

We allow multiple loaders to improve loading performance. There are a lot
of different strategies one could use when loading chunks. For example,
we tend to load chunks at a higher level prior to loading the chunks
at the ideal level. This gets "coverage" on the screen quickly, and then
the data can be refined by loading the ideal chunks.

One consideration is during rapid movement of the camera it's easy to clog
up the loader pool with workers loading chunks that have already moved out
of view. The
{class}`~napari.components.experimental.chunk._delay_queue.DelayQueue` was
created to help with this problem.

While we can't cancel a load if a worker has started working on it, we can
trivially cancel loads that are still in our delay queue. If the chunk goes
out of view, we cancel the load. If the user pauses for a bit, we initiate
the loads.

With multiple loaders we can delay the ideal chunks, but we can configure
zero delay for the higher levels. A single chunk from two levels up will
cover 16 ideal chunks. So immediately loading them is a good way to get
data on the screen quickly. When the camera stops moving the
{class}`~napari.components.experimental.chunk._pool.LoaderPool` for the
ideal layer will often be empty. So all of those workers can immediately
start loading the ideal chunks.

The ability to have multiple loaders was only recently added. We still need
to experiment to figure out the best configuration. And figure out how that
configuration needs to vary based on the latency of the data or other
considerations.

### Future work: Compatibility with the existing Image class

The focus for initial Octree development was Octree-specific behaviors and
infrastructure. Loading chunks asynchronously and rendering them as
individual tiles. One question we wanted to answer was will a Python/Vispy
implementation of Octree rendering be performant enough? Because if not, we
might need a totally different approach. It's not been fully proven out,
but it seems like the performance will be good enough, so the next step is
full compatibility with the existing
{class}`~napari.layers.Image` class.

The {class}`~napari.layers.image.experimental.octree_image.OctreeImage`
class is derived from {class}`~napari.layers.Image`, while
{class}`~napari._vispy.experimental.vispy_tiled_image_layer.VispyTiledImageLayer`
is derived from {class}`~napari._vispy.vispy_image_layer.VispyImageLayer`,
and
{class}`~napari._vispy.experimental.tiled_image_visual.TiledImageVisual` is
derived from the regular Vispy `ImageVisual` class. To bring full
{class}`~napari.layers.Image` capability to
{class}`~napari.layers.image.experimental.octree_image.OctreeImage` in most
cases we just need to duplicate what those base classes are doing, but do
it on a per-tile bases. Since there is no full image for them to operate
on. This might involve chaining to the base class or it could mean
duplicating that functionality somehow in the derived class.

Some {class}`~napari.layers.Image` functionality that needs to
be duplicated in Octree code:

#### Contrast limits and color transforms

The contrast limit code in Vispy's `ImageVisual` needs to be moved into
the tiled visual's
{meth}`~napari._vispy.experimental.tiled_image_visual.TiledImageVisual._build_texture`.
Instead operating on `self.data` it needs to transform tile's which are newly
being added to the visual. The color transform similarly needs to be per-tile.

#### Blending and opacity

It might be hard to get opacity working correctly for tiles where loads are
in progress. The way
{class}`~napari._vispy.experimental.tiled_image_visual.TiledImageVisual`
works today is the
{class}`~napari.layers.image.experimental._octree_loader.OctreeLoader`
potentially passes the visual tiles of various sizes, from different levels
of the Octree. The tiles are rendered on top of each other from largest
(coarsest level) to smallest (finest level). This is a nice trick so that
bigger tiles provide "coverage" for an area, while the smaller tiles add
detail only where that data has been loaded.

However, this breaks blending and opacity. We draw multiple tiles on top of
each other, so the image is blending with itself. One solution which is
kind of a big change is keep
{class}`~napari._vispy.experimental.tiled_image_visual.TiledImageVisual`
for the generic "tiled" case, but introduce a new `OctreeVisual` that
knows about the Octree. It can walk up and down the Octree chopping up
larger tiles to make sure we do not render anything on top of anything
else.

Until we do that, we could punt on making things look correct while loads
are in progress. We could even highlight the fact that a tile has not been
fully loaded (purposely making it look different until the data is fully
loaded). Aside from blending, this would address a common complaint with
tiled image viewers: you often can't tell if the data is still being
loaded. This could be a big issue for scientific uses, you don't want
people drawing the wrong conclusions from the data.

#### Time-series multiscale

To make time-series multiscale work should not be too hard. We just need to
correctly create a new
{class}`~napari.layers.image.experimental._octree_slice.OctreeSlice` every
time the slice changes.

The challenge will probably be performance. For starters we probably need
to stop creating the "extra" downsampled levels, as described in {ref}`future-work-atlas-2D`. We need to make sure constructing and
tearing down the Octree is fast enough, and make sure loads for the
previous slices are canceled and everything is cleaned up.


(future-work-atlas-2D)=
### Future work: Extending TextureAtlas2D

We could improve our
{class}`~napari._vispy.experimental.texture_atlas.TextureAtlas2D` class in
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
large root tiles. Graphics cards can handle pretty big textures. A layer
that's (100000, 100000) obviously needs to be broken into tiles, b¡ut a
layer that's (4096, 4096) really does not need to be broken into tiles.
That could be a single large tile.

Long term it would be nice if we did not have to support two image classes:
{class}`~napari.layers.Image` and
{class}`~napari.layers.image.experimental.octree_image.OctreeImage`.
Maintaining two code paths and two sets of visuals will become tiresome and
lead to discrepancies and bugs.

Instead, it would be nice if
{class}`~napari.layers.image.experimental.octree_image.OctreeImage` became
the only image class. One image class to rule them all. For that to happen,
though, we need to render small images just as efficiently as the
{class}`~napari.layers.Image` class does today. We do not want
Octree rendering to worsen cases which work well today. To keep today's
performance for smaller images we probably need to add support for variable
size tiles.

### Future work: Level-zero-only Octrees

In issue [#1300](https://github.com/napari/napari/issues/1300) it takes
1500ms to switch slices. There we are rendering a (16384, 16384) image that
is entirely in RAM. The delay is not from loading into RAM, it's already in
RAM, the delay is from transferring all that data to VRAM in one big gulp.

The image is not a multi-scale image. So can we turn it into a muli-scale
image? Generally we've found downsampling to create multi-scale image
layers is slow. So the question is how can we draw this large image without
hanging? One idea is we could create an Octree that only has a level zero
and no downsampled levels.

This is an option because chopping up a `NumPy` array into tiles is very
fast. This chopping up phase is really just creating a bunch of "views"
into the single existing array. So creating a level zero Octree should be
very fast. For there we can use our existing Octree code and our existing
{class}`~napari._vispy.experimental.vispy_tiled_image_visual.TiledImageVisual`
to transfer over one tile at a time without hurting the frame rate.

The insight here is our Octree code is really two things, one is an Octree
but two is a tiled or chunked image, basically a flat image chopped into a
grid of tiles. How would this look to the user? With this approach
switching slices would be similar to panning and zooming a multiscale
Octree image, you'd see the new tiles loading in over time, but the
framerate would not tank, and you could switch slices at any time.

### Future work: Caching

Basically no work has gone into caching or memory management for Octree
data. It's very likely there are leaks and extended usage will run out of
memory. This hasn't been addressed because using Octree for long periods of
time is just now becoming possible.

One caching issue is figuring out how to combine the `ChunkCache` with
Dasks's built-in caching. We probably want to keep the `ChunkCache` for
rendering non-Dask arrays? But when using Dask, we defer to its cache? We
certainly don't want to cache the data in both places.

Another issue is whether to cache `OctreeChunks` or tiles in the visual,
beyond just caching the raw data. If re-creating both is fast enough, the
simpler thing is evict them fully when a chunk falls out of view. And
re-create them if it comes back in view. It's simplest to keep nothing but
what we are currently drawing.

However if that's not fast enough, we could have a MRU cache of
`OctreeChunks` and tiles in VRAM, so that reviewing the same data is
nearly instant. This is adding complexity, but the performance might be
worth it.
