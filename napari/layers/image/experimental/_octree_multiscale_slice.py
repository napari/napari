"""OctreeMultiscaleSlice class.

For viewing one slice of a multiscale image using an octree.
"""
import logging
import math
from typing import Callable, List, Optional

import numpy as np

from ....components.experimental.chunk import ChunkRequest
from ....types import ArrayLike
from .._image_view import ImageView
from .octree import Octree
from .octree_chunk import OctreeChunk, OctreeLocation
from .octree_intersection import OctreeIntersection, OctreeView
from .octree_level import OctreeLevel, OctreeLevelInfo
from .octree_util import SliceConfig

LOGGER = logging.getLogger("napari.async.octree")


class OctreeMultiscaleSlice:
    """View a slice of an multiscale image using an octree."""

    def __init__(
        self,
        data,
        slice_config: SliceConfig,
        image_converter: Callable[[ArrayLike], ArrayLike],
    ):
        self.data = data

        self._slice_config = slice_config

        slice_id = id(self)
        self._octree = Octree(slice_id, data, slice_config)
        self._octree_level = self._octree.num_levels - 1

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
        return self._octree.levels[self._octree_level].info

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
            return self._octree_level

        # Find the right level automatically. Choose a level where the texels
        # in the octree tiles are around the same size as screen pixels.
        # We can do this smarter in the future, maybe have some hysterisis
        # so you don't "pop" to the next level as easily, so there is some
        # fudge factor or dead zone.
        ratio = view.data_width / view.canvas[0]

        if ratio <= 1:
            return 0  # Show the best we've got!

        # Choose the right level...
        return min(math.floor(math.log2(ratio)), self._octree.num_levels - 1)

    def get_visible_chunks(self, view: OctreeView) -> List[OctreeChunk]:
        """Return the chunks currently in view.

        Return
        ------
        List[OctreeChunk]
            The chunks which are visible in the given view.
        """
        intersection = self.get_intersection(view)

        if intersection is None:
            return []

        if view.auto_level:
            # Update our self._octree_level based on what level was automatically
            # selected by the intersection.
            self._octree_level = intersection.level.info.level_index

        # Return the chunks in this intersection.
        return intersection.get_chunks(id(self))

    def _get_octree_chunk(self, location: OctreeLocation):
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
            # This location in the octree is not a OctreeChunk. That's unexpected,
            # becauase locations are turned into OctreeChunk's when a load
            # is initiated. So this is an error, but log it and keep going.
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

        # OctreeChunk should no longer need to be loaded. We can probably
        # remove this check eventually, but for now to be sure.
        assert not self._get_octree_chunk(location).needs_load

        return True  # Chunk was loaded.
