"""OctreeLoader class.

Uses ChunkLoader to load data into OctreeChunks in the octree.
"""
import logging
from typing import List, Set

from ....components.experimental.chunk import (
    ChunkRequest,
    LayerRef,
    chunk_loader,
)
from ._chunk_set import ChunkSet
from .octree import Octree
from .octree_chunk import OctreeChunk, OctreeLocation

LOGGER = logging.getLogger("napari.octree.loader")
LOADER = logging.getLogger("napari.loader.futures")

# TODO_OCTREE make this a config. This is how many levels "up" we look
# for tiles to draw at levels above the ideal how. These tiles give
# us lots of coverage quickly, so we load and draw then even before
# the ideal level
NUM_ANCESTOR_LEVELS = 3


class OctreeLoader:
    """Load data into the OctreeChunks in the octree.

    The loader is given drawn_set, the chunks we are currently drawing, and
    ideal_chunks, the chunks which are in view at the desired level of the
    octree.

    The ideal level was chosen because its image pixels best match the
    screen pixels. Using higher resolution than that is okay, but it's
    wasted time and memory. Using lower resolution is better than nothing,
    but it's going to be blurrier than the ideal level.

    Our get_drawable_chunks() method iterates through the ideal_chunks
    choosing what chunks to load, in what order, and producing the set of
    chunks the visual should draw.

    Choosing what chunks to load and draw is the heart of octree rendering.
    We use the tree structure to find child or parent chunks, or chunks
    futher up the tree: ancestor chunks.

    The goal is to pretty quickly load all the ideal chunks, since that's
    what we really want to draw. But in the meantime we load and display
    chunks at lower or high resolutions. In some cases because they already
    loaded and even already being drawn. In other cases though we load
    chunk from high level because they provide "coverage" quickly.

    As you go up to higher levels from the ideal level, the chunks on those
    levels cover more and more chunks on the ideal level. As you go up
    levels they cover this number of ideal chunks: 4, 16, 64.

    The data from higher levels is blurry compared to the ideal level, but
    getting something "reasonable" on the screen quickly often leads to the
    best user experience. For example, even "blurry" data is often good
    enough for them to keep navigating, to keep panning and zooming looking
    for whatever they are looking for.

    Parameters
    ----------
    octree : Octree
        We are loading chunks for this octree.
    layer_ref : LayerRef
        A weak reference to the layer the octree lives in.

    Attributes
    ----------
    _octree : Octree
        We are loading chunks for this octree.
    _layer_ref : LayerRef
        A weak reference to the layer the octree lives in.
    """

    def __init__(self, octree: Octree, layer_ref: LayerRef):
        self._octree = octree
        self._layer_ref = layer_ref

    def get_drawable_chunks(
        self,
        drawn_set: Set[OctreeChunk],
        ideal_chunks: List[OctreeChunk],
        ideal_level: int,
    ) -> List[OctreeChunk]:
        """Return the chunks that should be drawn.

        The ideal chunks are within the bounds of the OctreeView, but they
        may or may not be in memory. We only return chunks which are in
        memory.

        Generally we want to draw the "best available" data. However, that
        data might not be at the ideal level. Sometimes we even load chunks
        at a higher level before loading the ideal chunks. To get
        "coverage" quickly.

        So we look in two directions:
        1) Up, to find a chunk at a higher (coarser) level.
        2) Down, to look for a drawable chunk at a lower (finer) level.

        The TiledImageVisual can draw overlapping tiles/chunks. For example
        suppose below B and C are ideal chunks, but B is drawable while C
        is not. We search up from C and find A.

        ----------
        |    A   |
        | --- ---|
        |  B | C |
        |---------

        TiledImageVisual will render A first, because it's at a higher
        level, and then B. So the visual will render B and A with B on top.
        The region defined by C is showing A, until C is ready to draw.

        Parameters
        ----------
        drawn_chunk_set : Set[OctreeChunk]
            The chunks which the visual is currently drawing.
        ideal_chunks : List[OctreeChunk]
            The chunks which are visible to the current view.

        Returns
        -------
        List[OctreeChunk]
            The chunks that should be drawn.
        """
        LOGGER.debug(
            "get_drawable_chunks: Starting with draw_set=%d ideal_chunks=%d",
            len(drawn_set),
            len(ideal_chunks),
        )

        # This is an ordered set. It's a set because many ideal chunks will
        # have the same ancestors, but we only want them in here once.
        seen = ChunkSet()

        # Permanent chunks are ones we always want to draw no matter where
        # the view is. For now this is just the root tile. These get loaded
        # first which is what we want.
        seen.add(self._get_permanent_chunks())

        # Now get coverage for every ideal chunk. This might include
        # the ideal chunk itself and/or chunks from other levels.
        for ideal_chunk in ideal_chunks:
            seen.add(self._get_coverage(ideal_chunk, drawn_set))

        # Add the ideal chunks AFTER all the coverage ones, we want to load
        # these after, because the coverage ones cover a much bigger area,
        # better to see them first, even though they are lower resolution.
        seen.add(ideal_chunks)

        # Cancel in-progress loads for any chunks we can no long see. When
        # panning or zooming rapidly, it's very common that chunks fall out
        # of view before the load was even started. We need to cancel those
        # loads or it will tie up the loader loading chunks we aren't even
        # going to display.
        self._cancel_unseen(seen)

        drawable = []

        # Load everything in seen if needed.
        for chunk in seen.chunks():
            # The ideal level is priority 0, 1 is one level above idea, etc.
            priority = chunk.location.level_index - ideal_level

            if chunk.in_memory:
                drawable.append(chunk)
            elif chunk.needs_load and self._load_chunk(chunk, priority):
                drawable.append(chunk)  # It was a sync load, ready to draw.

        # Useful for debugging but very spammy.
        # log_chunks("drawable", drawable)

        return drawable

    def _get_permanent_chunks(self) -> List[OctreeChunk]:
        """Get any permanent chunks we want to always draw.

        Right now it's just the root tile. We draw this so that we always
        have at least some minimal coverage when the camera moves to a new
        place. On a big enough dataset though when zoomed in we might be
        "inside" a single pixel of the root tile. So it's just providing a
        background color at that point.

        Returns
        -------
        List[OctreeChunk]
            Any extra chunks we should draw.
        """
        # We say create=True because the root is not part of the current
        # intersection. However since it's permanent once created and
        # loaded it should always be available. As long as we don't garbage
        # collect it!
        root_tile = self._octree.levels[-1].get_chunk(0, 0, create=True)
        return [root_tile]

    def _get_coverage(
        self, ideal_chunk: OctreeChunk, drawn_set: Set[OctreeChunk]
    ) -> List[OctreeChunk]:
        """Return the chunks to draw for this one ideal chunk.

        If the ideal chunk is already being drawn, we return it alone. It's
        all we need to draw to cover the chunk. If it's not being draw we
        look up down the tree to find what chunks we can to draw to "cover"
        this chunk.

        Note that drawn_set might be smaller than what get_drawable_chunks
        has been returning, because it only contains chunks that are
        actually got drawn to the screen. That are in VRAM.

        The visual might take time to load chunks into VRAM. So we might
        return the same chunks from get_drawable_chunks() many times
        in a row before it gets drawn. It might only one chunk per
        frame into VRAM, for example.

        Parameters
        ----------
        ideal_chunk : OctreeChunk
            The ideal chunk we'd like to draw.
        drawn_set : Set[OctreeChunk]
            The chunks which the visual is currently drawing.

        Returns
        -------
        List[OctreeChunk]
            The chunks that should be drawn to cover this one ideal chunk.
        """

        # If the ideal chunk is already being drawn, that's all we need,
        # there is no point in returning more than that.
        if ideal_chunk.in_memory and ideal_chunk in drawn_set:
            return [ideal_chunk]

        # Get alternates for this chunk, from other levels.
        family = self._get_family(ideal_chunk)

        ideal_level_index = ideal_chunk.location.level_index

        # For levels below the ideal level, we only keep an alternate if
        # it's already being drawn. This is usually when zooming out. The
        # alternates are "too small" but still look fine on screen.
        #
        # For levels above the ideal level, we will load and draw them. We
        # even sort so they get loaded and drawn *before* the ideal chunk.
        #
        # We do this because they provide coverage very quickly, and the
        # best user experience is to see imagery quickly even if not at the
        # ideal level.
        def keep_chunk(chunk) -> bool:
            lower_level = chunk.location.level_index < ideal_level_index

            if lower_level:
                return chunk in drawn_set

            return True  # Keep all higher level chunks.

        keep = [chunk for chunk in family if keep_chunk(chunk)]

        return keep

    def _get_family(self, ideal_chunk: OctreeChunk) -> List[OctreeChunk]:
        """Return chunks below and above this ideal chunk.

        Parameters
        ----------
        ideal_chunk : OctreeChunk
            Get children and parents of this chunk.

        Returns
        -------
        List[OctreeNode]
            Parents and children we should load and/or draw.
        """
        # Get any direct children which are in memory. Do not create
        # OctreeChunks or use children that are not already in memory
        # because it's better to create and load higher levels.
        children = self._octree.get_children(
            ideal_chunk, create=False, in_memory=True
        )

        # Get the parent and maybe more distant ancestors. Even if we have
        # all four children, we still consider loading and drawing these
        # because they will provide more coverage. They will cover the
        # ideal chunk plus more.
        ancestors = self._octree.get_ancestors(
            ideal_chunk, NUM_ANCESTOR_LEVELS, create=True
        )

        return children + ancestors

    def _load_chunk(self, octree_chunk: OctreeChunk, priority: int) -> None:
        """Load the data for one OctreeChunk.

        Parameters
        ----------
        octree_chunk : OctreeChunk
            Load the data for this chunk.
        """
        # We only want to load a chunk if it's not already in memory, if a
        # load was not started on it.
        assert not octree_chunk.in_memory
        assert not octree_chunk.loading

        # The ChunkLoader takes a dict of chunks that should be loaded at
        # the same time. Today we only ever ask it to a load a single chunk
        # at a time. In the future we might want to load multiple layers at
        # once, so they are in sync, or load multiple locations to bundle
        # things up for efficiency.
        chunks = {'data': octree_chunk.data}

        # Mark that this chunk is being loaded.
        octree_chunk.loading = True

        # Create the ChunkRequest and load it with the ChunkLoader.
        request = ChunkRequest(octree_chunk.location, chunks, priority)
        satisfied_request = chunk_loader.load_request(request)

        if satisfied_request is None:
            # An async load was initiated. The load will probably happen in a
            # worker thread. When the load completes QtChunkReceiver will call
            # OctreeImage.on_chunk_loaded() with the data.
            return False

        # The load was synchronous. Some situations were the
        # ChunkLoader loads synchronously:
        #
        # 1) The force_synchronous config option is set.
        # 2) The data already was an ndarray, there's nothing to "load".
        # 3) The data is Dask or similar, but based on past loads it's
        #    loading so quickly that we decided to load it synchronously.
        # 4) The data is Dask or similar, but we already loaded this
        #    exact chunk before, so it was in the cache.
        #

        # Whatever the reason, the data is now ready to draw.
        octree_chunk.data = satisfied_request.chunks.get('data')

        # The chunk has been loaded, it's now a drawable chunk.
        assert octree_chunk.in_memory
        return True

    def _cancel_unseen(self, seen: ChunkSet) -> None:
        """Cancel in-progress loads not in the seen set.

        Parameters
        ----------
        seen : ChunkSet
            The set of chunks the loader can see.
        """

        def _should_cancel(chunk_request: ChunkRequest) -> bool:
            """Cancel if we are no longer seeing this location."""
            return not seen.has_location(chunk_request.location)

        cancelled = chunk_loader.cancel_requests(_should_cancel)

        for request in cancelled:
            self._on_cancel_request(request.location)

    def _on_cancel_request(self, location: OctreeLocation) -> None:
        """Request for this location was cancelled.

        Parameters
        ----------
        location : OctreeLocation
            Set that this chunk is no longer loading.
        """
        # Get chunk for this location, don't create the chunk, but it ought
        # to be there since there was a load in progress.
        chunk: OctreeChunk = self._octree.get_chunk_at_location(
            location, create=False
        )

        if chunk is None:
            LOADER.error("_cancel_load: Chunk did not exist %s", location)
            return

        # Chunk is no longer loading.
        chunk.loading = False
