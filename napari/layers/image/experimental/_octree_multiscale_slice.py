"""OctreeMultiscaleSlice class.

For viewing one slice of a multiscale image using an octree.
"""
import logging
import math
from typing import Callable, List, Optional, Set

import numpy as np

from ....components.experimental.chunk import ChunkRequest, LayerRef
from ....types import ArrayLike
from .._image_view import ImageView
from ._octree_chunk_loader import OctreeChunkLoader
from .octree import Octree
from .octree_chunk import OctreeChunk, OctreeChunkKey, OctreeLocation
from .octree_intersection import OctreeIntersection, OctreeView
from .octree_level import OctreeLevel, OctreeLevelInfo
from .octree_util import SliceConfig

LOGGER = logging.getLogger("napari.async.octree")


class OctreeMultiscaleSlice:
    """View a slice of an multiscale image using an octree.

    Parameters
    ----------
    data
        The multi-scale data.
    slice_config : SliceConfig
        The base shape and other info.
    image_converter : Callable[[ArrayLike], ArrayLike]
        For converting to displaying data.

    Attributes
    ----------
    _loader : OctreeChunkLoader
        Uses the napari ChunkLoader to load OctreeChunks.

    """

    def __init__(
        self,
        data,
        layer_ref: LayerRef,
        slice_config: SliceConfig,
        image_converter: Callable[[ArrayLike], ArrayLike],
    ):
        self.data = data
        self._layer_ref = layer_ref
        self._slice_config = slice_config

        slice_id = id(self)
        self._octree = Octree(slice_id, data, slice_config)

        # Note that self._octree might have more levels than len(data) because
        # in some cases we add on extra levels. We add on levels until the
        # root tile consists of only a single tile.
        self._octree_level = self._octree.num_levels - 1

        self._loader: OctreeChunkLoader = OctreeChunkLoader(
            self._octree, layer_ref
        )

        thumbnail_image = np.zeros(
            (64, 64, 3)
        )  # blank until we have a real one
        self.thumbnail: ImageView = ImageView(thumbnail_image, image_converter)

    @property
    def octree_level(self) -> int:
        """The current octree level.

        Return
        ------
        int
            The current octree level.
        """
        return self._octree_level

    @octree_level.setter
    def octree_level(self, level: int) -> None:
        """Set the octree level we are viewing.

        Parameters
        ----------
        level : int
            The new level to display.
        """
        num_levels = self._octree.num_levels
        if level < 0 or level >= num_levels:
            raise ValueError(
                f"Octree level {level} is not in range(0, {num_levels})"
            )

        self._octree_level = level

    @property
    def loaded(self) -> bool:
        """Return True if the data has been loaded.

        Because octree multiscale is async, we say we are loaded up front even
        though none of our chunks/tiles might be loaded yet.
        """
        return self.data is not None

    @property
    def octree_level_info(self) -> Optional[OctreeLevelInfo]:
        """Return information about the current octree level.

        Return
        ------
        Optional[OctreeLevelInfo]
            Information about current octree level, if there is one.
        """
        if self._octree is None:
            return None

        try:
            return self._octree.levels[self.octree_level].info
        except IndexError as exc:
            index = self.octree_level
            num_levels = len(self._octree.levels)
            raise IndexError(
                f"Octree level {index} is not in range(0, {num_levels})"
            ) from exc

    def get_intersection(self, view: OctreeView) -> OctreeIntersection:
        """Return this view's intersection with the octree.

        Parameters
        ----------
        view : OctreeView
            Intersect this view with the octree.

        Return
        ------
        OctreeIntersection
            The view's intersection with the octree.
        """
        level = self._get_auto_level(view)
        return OctreeIntersection(level, view)

    def _get_auto_level(self, view: OctreeView) -> OctreeLevel:
        """Get the automatically selected octree level for this view.

        Parameters
        ----------
        view : OctreeView
            Get the OctreeLevel for this view.

        Return
        ------
        OctreeLevel
            The automatically chosen OctreeLevel.
        """
        index = self._get_auto_level_index(view)
        if index < 0 or index >= self._octree.num_levels:
            raise ValueError(f"Invalid octree level {index}")

        return self._octree.levels[index]

    def _get_auto_level_index(self, view: OctreeView) -> int:
        """Get the automatically selected octree level index for this view.

        Parameters
        ----------
        view : OctreeView
            Get the octree level index for this view.

        Return
        ------
        int
            The automatically chosen octree level index.
        """
        if not view.auto_level:
            # Return current level, do not update it.
            return self.octree_level

        # Find the right level automatically. Choose a level where the texels
        # in the octree tiles are around the same size as screen pixels.
        # We can do this smarter in the future, maybe have some hysterisis
        # so you don't "pop" to the next level as easily, so there is some
        # fudge factor or dead zone.
        ratio = view.data_width / view.canvas[0]

        if ratio <= 1:
            return 0  # Show the best we've got!

        # Choose the right level...
        max_level = self._octree.num_levels - 1
        return min(math.floor(math.log2(ratio)), max_level)

    def get_drawable_chunks(
        self, drawn_chunk_set: Set[OctreeChunkKey], view: OctreeView
    ) -> List[OctreeChunk]:
        """Get the chunks that should be drawn to depict this view.

        Parameters
        ----------
        drawn_chunk_set : Set[OctreeChunkKey]
            The chunks that are currently being drawn by the visual.
        view : OctreeView
            Get the chunks for this view.

        Return
        ------
        List[OctreeChunk]
            The chunks to draw.
        """
        # Get the ideal chunks, the ones best match the current screen
        # resolution.
        ideal_chunks = self._get_ideal_chunks(view)

        layer_key = self._layer_ref.layer_key

        # Let the loader decide what chunks should be drawn. It will
        # only return chunks which are fully loaded and read to be drawn.
        #
        # It might return chunks from a higher or lower level than the
        # ideal level. Also it might initiate async loads so that more
        # chunks will be drawable in the near future.
        return self._loader.get_drawable_chunks(
            drawn_chunk_set, ideal_chunks, layer_key
        )

    def _get_ideal_chunks(self, view: OctreeView) -> List[OctreeChunk]:
        """Get the ideal chunks we want to draw for this view.

        The call to get_intersection() will chose the appropriate level of
        the octree to intersect, and then return all the chunks within the
        intersection with that level.

        These are the "ideal" chunks because they are at the level whose
        resolution best matches the current screen resolution.

        Drawing chunks at a lower level than this will work fine, but it's
        a waste in that those chunks will just be downsampled by the card.
        You won't see any "extra" resolution at all. The card can do this
        super fast, so the issue not such much speed as it is RAM and VRAM.

        For example, suppose we want to draw 40 ideal chunks at level N,
        and the chunks are (256, 256, 3) with dtype uint8. That's around
        8MB.

        If instead we draw lower levels than the ideal, the number of
        chunks and storage goes up quickly:

        Level (N - 1) is 160 chunks = 32M
        Level (N - 2) is 640 chunks = 126M
        Level (N - 3) is 2560 chunks = 503M

        In the opposite direction, drawing chunks from a higher, the number
        of chunks and storage goes down quickly. The only issue there is
        visual quality, the imagery might look blurry.

        Paramaters
        ----------
        view : OctreeView
            Return chunks that are within this view.

        Return
        ------
        List[OctreeChunk]
            The chunks which are visible in the given view.
        """
        intersection = self.get_intersection(view)

        if intersection is None:
            return []  # No visible chunks.

        # If we are choosing the level automatically, then update our level
        # with the level chosen for the intersection.
        if view.auto_level:
            self.octree_level = intersection.level.info.level_index

        # Return all of the chunks in this intersection, creating chunks if
        # they don't already exist. We create them because we want to load
        # the data in these locations.
        return intersection.get_chunks(create=True)

    def _get_octree_chunk(self, location: OctreeLocation) -> OctreeChunk:
        """Return the OctreeChunk at his location.

        Parameters
        ----------
        location : OctreeLocation
            Return the chunk at this location.

        Return
        ------
        OctreeChunk
            The returned chunk.
        """
        level = self._octree.levels[location.level_index]
        return level.get_chunk(location.row, location.col)

    def on_chunk_loaded(self, request: ChunkRequest) -> None:
        """An asynchronous ChunkRequest was loaded.

        Override Image.on_chunk_loaded() fully.

        Parameters
        ----------
        request : ChunkRequest
            This request was loaded.
        """
        location = request.key.location
        if location.slice_id != id(self):
            # There was probably a load in progress when the slice was changed.
            # The original load finished, but we are now showing a new slice.
            # Don't consider it error, just ignore the chunk.
            LOGGER.debug("on_chunk_loaded: wrong slice_id: %s", location)
            return False  # Do not load.

        octree_chunk = self._get_octree_chunk(location)
        if not isinstance(octree_chunk, OctreeChunk):
            # This location in the octree does not contain an OctreeChunk.
            # That's unexpected, becauase locations are turned into
            # OctreeChunk's when a load is initiated. So this is an error,
            # but log it and keep going, maybe some transient weirdness.
            LOGGER.error(
                "on_chunk_loaded: missing OctreeChunk: %s", octree_chunk
            )
            return False  # Do not load.

        # Looks good, we are loading this chunk.
        LOGGER.debug("on_chunk_loaded: loading %s", octree_chunk)

        # Get the data from the request.
        incoming_data = request.chunks.get('data')

        # Loaded data should always be an ndarray.
        assert isinstance(incoming_data, np.ndarray)

        # Shove the request's ndarray into the octree's OctreeChunk. This octree
        # chunk now has an ndarray as its data, and it can be rendered.
        octree_chunk.data = incoming_data

        # Now needs_load should be false, since this OctreeChunk was
        # loaded. We can probably remove this check eventually, but for now
        # to be sure.
        assert not self._get_octree_chunk(location).needs_load

        return True  # Chunk was loaded.
