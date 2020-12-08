"""OctreeChunkLoader class.

Uses ChunkLoader to load data into OctreeChunks in the octree.
"""
import logging
from typing import List, Set

from ....components.experimental.chunk import LayerKey, LayerRef, chunk_loader
from .octree_chunk import OctreeChunk, OctreeChunkKey

LOGGER = logging.getLogger("napari.async.octree")


class OctreeChunkLoader:
    """Load data into OctreeChunks in the octree.

    Parameters
    ----------
    layer_ref : LayerRef
        A weak reference to the layer we are loading chunks for.

    Attributes
    ----------
    _last_visible : Set[OctreeChunkKey]
        Chunks we saw last frame, so we can recognize chunks which have just
        come into view.
    """

    def __init__(self, layer_ref: LayerRef):
        self._layer_ref = layer_ref
        self._last_visible: Set[OctreeChunkKey] = set()

    def get_drawable_chunks(
        self, visible: List[OctreeChunk], layer_key: LayerKey
    ) -> List[OctreeChunk]:
        """Return the chunks that should be drawn.

        Visible chunks are within the bounds of the OctreeView, but those
        chunks may or may not be drawable. Drawable chunks are typically
        ones that were fully in memory to start, or have been
        asynchronously loaded so their data is now in memory.

        This routine might return drawable chunks for levels other than
        the target level we are drawing. This is called multi-level rendering
        and it's core feature of quadtree/octree rendering.

        Background
        -----------
        Generally we want to draw the "best available" data. The target
        level was chosen because its image pixels best match the screen
        pixels. So if available we always prefer to draw chunks at the
        target level.

        However we strongly want to avoid drawing nothing. So if a chunk is
        not available in the target level, we search of a suitable substitute.
        There are just two directions to search:

        1) Look for a drawable chunk at a higher (coarser) level.
        2) Look for a drawable chunk at a lower (finer) level.

        A parent chunk, at a higher level, will cover more than just the
        missing target level chunk. While a child chunk, at a lower level,
        will only cover a fraction of the missing target level chunk.

        The TiledImageVisual can draw overlapping tiles/chunks. For example
        suppose below B and C are in the target level. But in this case
        B is drawable but C is not. We search up from C and find A.
        ----------
        |    A   |
        | --- ---|
        |  B | C |
        |---------
        We return B and A as drawable chunks. TiledImageVisual will render
        A first and then B. So looking at the region coverede by A, the 1/4
        occupied by B will be at the target resolution, the remaining 3/4
        will be at a a lower resoution.

        In many cases drawing one level lower resolution than the target will
        be barely noticeable by the user. And once the chunk is loaded it
        will update to full resolution.

        Parameters
        ----------
        visible : List[OctreeChunk]
            The chunks which are visible to the current view.
        layer_key : LayerKey
            The layer we loading chunks into.

        Return
        ------
        List[OctreeChunk]
            The chunks that can be drawn.
        """
        # Create a set for fast membership testing.
        visible_set = set(octree_chunk.key for octree_chunk in visible)

        # Remove chunks from self._last_visible if they are no longer
        # in the visible set. If they are no longer in view.
        for key in list(self._last_visible):
            if key not in visible_set:
                self._last_visible.remove(key)

        drawable = []  # Create this list of drawable chunks.

        # Get drawables for every visible chunks. This might be the chunk itself
        # or it might be stand-ins from other levels.
        for i, octree_chunk in enumerate(visible):
            chunk_drawables = self._get_drawables(octree_chunk, layer_key)
            drawable.extend(chunk_drawables)

        # Update our _last_visible set with what is in view.
        for octree_chunk in drawable:
            self._last_visible.add(octree_chunk.key)

        return drawable

    def _load_chunk(
        self, octree_chunk: OctreeChunk, layer_key: LayerKey
    ) -> None:
        """Load the data for one OctreeChunk.

        Parameters
        ----------
        octree_chunk : OctreeChunk
            Load the data for this chunk.
        layer_key : LayerKey
            The key for layer we are loading the data for.
        """
        # Key that points to a specific location in the octree.
        key = OctreeChunkKey(layer_key, octree_chunk.location)

        # We only load one chunk per request right now, so we just
        # call it 'data'.
        chunks = {'data': octree_chunk.data}

        # Mark that a load is in progress for this OctreeChunk. So
        # we don't initiate a second load for one reason.
        octree_chunk.loading = True

        # Create the ChunkRequest and load it with the ChunkLoader.
        request = chunk_loader.create_request(self._layer_ref, key, chunks)
        satisfied_request = chunk_loader.load_chunk(request)

        if satisfied_request is None:
            # An async load as initiated. The load will probably happen
            # in a worker thread. When the load completes QtChunkReceiver
            # will call OctreeImage.on_chunk_loaded() with the data.
            return False

        # The load was sync so it's already done, some situations were
        # the ChunkLoader loads synchronously:
        #
        # 1) Its force_synchronous config option is set.
        # 2) The data already is an ndarray, there's nothing to load.
        # 3) The data is Dask or similar, but based on past loads it's
        #    loading so quickly, we decided to load it synchronously.
        # 4) The data is Dask or similar, but we already loaded this
        # exact chunk before, so it was in the cache.
        #
        # Whatever the reason, we can insert the data into the octree and
        # we will draw it this frame.
        octree_chunk.data = satisfied_request.chunks.get('data')
        return True

    def _get_drawables(
        self, octree_chunk: OctreeChunk, layer_key: LayerKey
    ) -> bool:
        """Return True if this chunk is ready to be drawn.

        Parameters
        ----------
        octree_chunk : OctreeChunk
            Prepare to draw this chunk.

        Return
        ------
        bool
            True if this chunk can be drawn.
        """
        # If the cache is disabled, and this chunk just came into view,
        # the we nuke the contents. This forces a re-load of the data.
        #
        # In the future we might strip every chunk that goes out of
        # view. Then this will not be needed.
        if not chunk_loader.cache.enabled:
            new_in_view = octree_chunk.key not in self._last_visible
            if new_in_view and octree_chunk.in_memory:
                octree_chunk.clear()

        # If the chunk is fully in memory, then it's drawable.
        if octree_chunk.in_memory:
            return [octree_chunk]  # Draw the chunk itself.

        # If the chunk is loading, it's not drawable yet.
        if octree_chunk.loading:
            return self._get_replacements(octree_chunk)

        # The chunk is not in memory and is not being loaded, but it is
        # in view. So try loading it.
        sync_load = self._load_chunk(octree_chunk, layer_key)

        # If the chunk was loaded synchronously, we can draw it now.
        if sync_load:
            return [octree_chunk]  # Draw the chunk itself.

        # Otherwise, an async load as initiated, and sometime later
        # OctreeImage.on_chunk_loaded will be called with the chunk's
        # loaded data. But we can't draw it now.
        return self._get_replacements(octree_chunk)

    def _get_replacements(
        self, octree_chunk: OctreeChunk
    ) -> List[OctreeChunk]:
        """Return chunks we are draw in place of the given chunk.

        Parameters
        ----------
        octree_chunk : OctreeChunk
            Get chunks to draw in place of this one.

        Return
        ------
        List[OctreeNode]
            Draw this chunks as a replacement.
        """
        return []  # nothing yet
