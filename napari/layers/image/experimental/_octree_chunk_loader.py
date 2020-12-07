"""OctreeChunkLoader class.

Uses ChunkLoader to load OctreeChunk's in the octree.
"""
import logging
from typing import List

from ....components.experimental.chunk import LayerKey, LayerRef, chunk_loader
from .octree_chunk import OctreeChunk, OctreeChunkKey

LOGGER = logging.getLogger("napari.async.octree")


class OctreeChunkLoader:
    """Load chunks for the octree.

    Parameters
    ----------
    layer_ref : LayerRef
        A weak reference to the layer we are loading chunks for.
    """

    def __init__(self, layer_ref: LayerRef):
        self._layer_ref = layer_ref
        self._last_visible = set()

    def get_drawable_chunks(
        self, visible: List[OctreeChunk], layer_key: LayerKey
    ) -> List[OctreeChunk]:
        """Return the chunks that can be drawn, of the visible chunks.

        Drawable chunks are typically the ones that are fully in memory.
        Either they were always in memory, or they are now in memory after
        having been loaded asynchronously by the hunkLoader.

        Parameters
        ----------
        visible : List[OctreeChunk]
            The chunks which are visible to the camera.

        Return
        ------
        List[OctreeChunk]
            The chunks that can be drawn.
        """
        visible_set = set(octree_chunk.key for octree_chunk in visible)

        # Remove any chunks from our self._last_visible set which are no
        # longer in view.
        for key in list(self._last_visible):
            if key not in visible_set:
                self._last_visible.remove(key)

        count = len(visible)

        def _log(i, label, chunk):
            LOGGER.debug(
                "Visible Chunk: %d of %d -> %s: %s", i, count, label, chunk
            )

        drawable = []  # TODO_OCTREE combine list/set
        visible_set = set()
        for i, octree_chunk in enumerate(visible):

            if not chunk_loader.cache.enabled:
                new_in_view = octree_chunk.key not in self._last_visible
                if new_in_view and octree_chunk.in_memory:
                    # Not using cache, so if this chunk just came into view
                    # clear it out, so it gets reloaded.
                    octree_chunk.clear()

            if octree_chunk.in_memory:
                # The chunk is fully in memory, we can view it right away.
                # _log(i, "ALREADY LOADED", octree_chunk)
                drawable.append(octree_chunk)
                visible_set.add(octree_chunk.key)
            elif octree_chunk.loading:
                # The chunk is being loaded, do not view it yet.
                _log(i, "LOADING:", octree_chunk)
            else:
                # The chunk is not in memory and is not being loaded, so
                # we are going to load it.
                sync_load = self._load_chunk(octree_chunk, layer_key)
                if sync_load:
                    # The chunk was loaded synchronously. Either it hit the
                    # cache, or it's fast-loading data. We can draw it now.
                    _log(i, "SYNC LOAD", octree_chunk)
                    drawable.append(octree_chunk)
                    visible_set.add(octree_chunk.key)
                else:
                    # An async load was initiated, sometime later our
                    # self._on_chunk_loaded method will be called.
                    _log(i, "ASYNC LOAD", octree_chunk)

        # Update our _last_visible set with what is in view.
        for octree_chunk in drawable:
            self._last_visible.add(octree_chunk.key)

        return drawable

    def _load_chunk(
        self, octree_chunk: OctreeChunk, layer_key: LayerKey
    ) -> None:

        key = OctreeChunkKey(layer_key, octree_chunk.location)

        chunks = {'data': octree_chunk.data}

        octree_chunk.loading = True

        # Create the ChunkRequest and load it with the ChunkLoader.
        request = chunk_loader.create_request(self._layer_ref, key, chunks)

        satisfied_request = chunk_loader.load_chunk(request)

        if satisfied_request is None:
            return False  # Load was async.

        # Load was sync so we can insert the data into the octree
        # and we will draw it this frame.
        octree_chunk.data = satisfied_request.chunks.get('data')
        return True
