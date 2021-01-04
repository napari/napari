.. _rendering:

Napari Rendering
================

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
