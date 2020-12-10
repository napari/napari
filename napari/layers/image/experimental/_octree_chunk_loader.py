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

        # Get the ideal checks or alternates, plus any extra chunks.
        for octree_chunk in ideal_chunks:
            drawable.extend(
                self._get_ideal_chunks(
                    drawn_chunk_set, octree_chunk, layer_key
                )
            )
            drawable.extend(self._get_extra_chunks())

        return drawable

    def _get_extra_chunks(self) -> List[OctreeChunk]:
        """Get any extra chunks we want to draw.

        Right now it's just the root tile. We draw this so that we always
        have at least some minimal coverage when the camera moves to a new
        place.

        Return
        ------
        List[OctreeChunk]
            Any extra chunks we should draw.
        """
        # We create it because it's not part of the intersection, it's just
        # something extra we want to draw.
        root_tile = self._octree.levels[-1].get_chunk(0, 0, create=True)
        return [root_tile]

    def _get_ideal_chunks(
        self,
        drawn_chunk_set: Set[OctreeChunkKey],
        ideal_chunk: OctreeChunk,
        layer_key: LayerKey,
    ) -> List[OctreeChunk]:
        """Return the chunks to draw for the given ideal chunk.

        If ideal chunk is already being drawn, we return just it.

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
        # If the ideal chunk is already being drawn, that's all we need.
        if ideal_chunk.in_memory and ideal_chunk.key in drawn_chunk_set:
            return [ideal_chunk]

        # If the ideal chunk has not be loaded at all yet, kick off a load.
        # This might happen sync or async. If sync then we'll see that
        # it's in memory below.
        if ideal_chunk.needs_load:
            self._load_chunk(ideal_chunk, layer_key)

        # Get the alternates for this chunk, from other levels.
        alternates = self._get_alternates(ideal_chunk, drawn_chunk_set)

        # Get which of these are currently in VRAM being drawn. Only these
        # can be shown instantly. Typically if we are zooming in, the drawn
        # alternates are the parent ones that were getting too big. While
        # if we are zooming out they are the children that were getting too
        # small.
        drawn = [chunk for chunk in alternates if chunk.key in drawn_chunk_set]

        # If the ideal chunk is in memory, we only want to send drawn
        # alternates. Because non-drawn alternates would take just as long
        # to get into VRAM as the ideal chunk.
        if ideal_chunk.in_memory:
            return [ideal_chunk] + drawn

        # Keep drawing these until the ideal chunk has loaded.
        return drawn

    def _get_alternates(
        self, ideal_chunk: OctreeChunk, drawn_chunk_set
    ) -> List[OctreeChunk]:
        """Return the chunks we could draw in place of the ideal chunk.

        Parameters
        ----------
        ideal_chunk : OctreeChunk
            Get chunks we could draw in place of this one.

        Return
        ------
        List[OctreeNode]
            We could draw these chunks.
        """
        # Get any direct children.
        alternates = self._octree.get_children(ideal_chunk)

        # Get a parent or a more distance ancestor.
        ancestor = self._octree.get_nearest_ancestor(ideal_chunk)
        if ancestor is not None:
            alternates.append(ancestor)

        return alternates

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
