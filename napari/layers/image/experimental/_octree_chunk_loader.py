"""OctreeChunkLoader class.

Uses ChunkLoader to load data into OctreeChunks in the octree.
"""
import logging
from typing import List, Set

from ....components.experimental.chunk import LayerKey, LayerRef, chunk_loader
from .octree import Octree
from .octree_chunk import OctreeChunk, OctreeChunkKey

LOGGER = logging.getLogger("napari.async.octree")


class OctreeChunkLoader:
    """Load data into OctreeChunks in the octree.

    Parameters
    ----------
    layer_ref : LayerRef
        A weak reference to the layer we are loading chunks for.
    """

    def __init__(self, octree: Octree, layer_ref: LayerRef):
        self._octree = octree
        self._layer_ref = layer_ref

    def get_drawable_chunks(
        self,
        drawn_chunk_set: Set[OctreeChunkKey],
        ideal_chunks: List[OctreeChunk],
        layer_key: LayerKey,
    ) -> List[OctreeChunk]:
        """Return the chunks that should be drawn.

        The ideal chunks are within the bounds of the OctreeView, but those
        chunks may or may not be drawable. Drawable chunks are typically
        ones that were fully in memory to start, or have been
        asynchronously loaded so their data is now in memory.

        This routine might return drawable chunks for levels other than the
        ideal level. This is called multi-level rendering and it's core
        feature of quadtree/octree rendering.

        Background
        -----------
        Generally we want to draw the "best available" data. The ideal
        level was chosen because its image pixels best match the screen
        pixels. So if loaded, we prefer drawing the ideal chunks.

        However, we strongly want to avoid drawing nothing. So if an ideal
        chunk is not available, we search for a suitable substitute. There
        are just two directions to search:

        1) Up, look for a drawable chunk at a higher (coarser) level.
        2) Down, look for a drawable chunk at a lower (finer) level.

        A parent chunk, at a higher level, will cover more than just the
        missing ideal chunk. While a child chunk, at a lower level, will
        only cover a fraction of the missing ideal chunk. And so on
        recursively through multiple levels.

        The TiledImageVisual can draw overlapping tiles/chunks. For example
        suppose below B and C are ideal chunks, but B is drawable while C
        is not. We search up from C and find A.

        ----------
        |    A   |
        | --- ---|
        |  B | C |
        |---------

        We return B and A as drawable chunks. TiledImageVisual will render
        A first and then B. So looking at the region covered by A, the 1/4
        occupied by B will be at the ideal resolution, while the remaining
        3/4 will be at a lower resoution.

        In many cases drawing one level lower resolution than the target will
        be barely noticeable by the user. And once the chunk is loaded it
        will update to full resolution.

        Drawing one level too high should not be noticeable at all by the
        user, it just means the card is doing the downsampling. The only
        downside there is drawing lots of tiles might be slow and it will
        hold more data in memory than is necessary.

        Parameters
        ----------
        drawn_chunk_set : Set[OctreeChunkKey]
            The chunks which the visual is currently drawing.
        ideal_chunks : List[OctreeChunk]
            The chunks which are visible to the current view.
        layer_key : LayerKey
            The layer we loading chunks into.

        Return
        ------
        List[OctreeChunk]
            The chunks that should be drawn.
        """
        drawable = []  # Create this list of drawable chunks.

        # Get drawables for every ideal chunk. This will be the chunk itself
        # if it's drawable. Otherwise it might be some number of substitute
        # chunks from higher or lower levels.
        for octree_chunk in ideal_chunks:
            chunk_drawables = self._get_drawables(
                drawn_chunk_set, octree_chunk, layer_key
            )
            drawable.extend(chunk_drawables)

        print(f"num_drawable = {len(drawable)}")
        return drawable

    def _get_drawables(
        self,
        drawn_chunk_set: Set[OctreeChunkKey],
        ideal_chunk: OctreeChunk,
        layer_key: LayerKey,
    ) -> bool:
        """Return True if this chunk is ready to be drawn.

        Parameters
        ----------
        drawn_chunk_set : Set[OctreeChunkKey]
            The chunks which the visual is currently drawing.
        ideal_chunk : OctreeChunk
            The ideal chunk we'd like to draw.
        layer_key : LayerKey
            The layer we loading chunks into.

        Return
        ------
        bool
            True if this chunk can be drawn.
        """
        # If the chunk is fully in memory, then it's drawable.
        if ideal_chunk.in_memory:
            # If it's being drawn already, then keep drawing it.
            if ideal_chunk.key in drawn_chunk_set:
                print(f"in_memory and draw: level {ideal_chunk.location}")
                return [ideal_chunk]

            # It's in memory but it's NOT being drawn yet. Maybe it has not
            # been paged into VRAM. Draw both the ideal chunk AND
            # substitutes until it starts being drawn. We draw both so
            # the visual keeps trying to add the idea one.
            print(f"in_memory and not drawn: level {ideal_chunk.location}")
            return [ideal_chunk] + self._get_substitutes(ideal_chunk)

        # The chunk is not in memory. If it's being loaded, draw
        # substitutes until the load finishes.
        if ideal_chunk.loading:
            return self._get_substitutes(ideal_chunk)

        # The chunk is not in memory and is not being loaded, so try
        # loading it now. We'd like to draw this chunk!
        sync_load = self._load_chunk(ideal_chunk, layer_key)

        # If the chunk was loaded synchronously, we can draw it now.
        if sync_load:
            return [ideal_chunk]

        # Otherwise, an async load was initiated, and sometime later
        # OctreeImage.on_chunk_loaded will be called with the chunk's
        # loaded data. But we can't draw it now.
        return self._get_substitutes(ideal_chunk)

    def _get_substitutes(self, ideal_chunk: OctreeChunk) -> List[OctreeChunk]:
        """Return the chunks we should draw in place of the ideal chunk.

        Parameters
        ----------
        ideal_chunk : OctreeChunk
            Get chunks to draw in place of this one.

        Return
        ------
        List[OctreeNode]
            Draw these chunks in place of the ideal one.
        """
        print(f"_get_substitute: ideal={ideal_chunk.location}")
        # TODO_OCTREE: This is just the very start of the search algorithm
        # as a test. This will be extended soon.
        parent_chunk = self._octree.get_parent(ideal_chunk)
        if parent_chunk is None:
            print("_get_substitute: found none")
            return []  # No immediate parent
        print(f"_get_substitute: found {parent_chunk.location}")
        return [parent_chunk]

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
        # Get a key that points to this specific location in the octree.
        key = OctreeChunkKey(layer_key, octree_chunk.location)

        # The ChunkLoader takes a dict of chunks that should be loaded at
        # the same time. We only ever ask for one chunk right now, so just
        # call it 'data'.
        chunks = {'data': octree_chunk.data}

        # Mark that a load is in progress for this OctreeChunk. We don't
        # want to initiate a second load for the same chunk.
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
        # 1) The force_synchronous config option is set.
        # 2) The data already was an ndarray, there's nothing to "load".
        # 3) The data is Dask or similar, but based on past loads it's
        #    loading so quickly that we decided to load it synchronously.
        # 4) The data is Dask or similar, but we already loaded this
        # exact chunk before, so it was in the cache.
        #
        # Whatever the reason, the data is now ready to draw.
        octree_chunk.data = satisfied_request.chunks.get('data')
        return True
