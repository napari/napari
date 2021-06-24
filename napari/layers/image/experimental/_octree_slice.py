"""OctreeSlice class.

For viewing one slice of a multiscale image using an octree.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

import numpy as np

from ....utils.translations import trans
from ._octree_loader import OctreeLoader
from .octree import Octree
from .octree_intersection import OctreeIntersection, OctreeView
from .octree_level import OctreeLevel, OctreeLevelInfo
from .octree_util import OctreeMetadata

LOGGER = logging.getLogger("napari.octree.slice")

if TYPE_CHECKING:
    from ....components.experimental.chunk import (
        ChunkRequest,
        LayerRef,
        OctreeLocation,
    )
    from .octree_chunk import OctreeChunk


class OctreeSlice:
    """A viewed slice of a multiscale image using an octree.

    Parameters
    ----------
    data
        The multi-scale data.
    layer_ref : LayerRef
        Reference to the layer containing the slice.
    meta : OctreeMetadata
        The base shape and other info.

    Attributes
    ----------
    loader : OctreeLoader
        Uses the napari ChunkLoader to load OctreeChunks.
    """

    def __init__(
        self,
        data,
        layer_ref: LayerRef,
        meta: OctreeMetadata,
    ):
        self.data = data
        self._meta = meta

        slice_id = id(self)
        self._octree = Octree(slice_id, data, meta)

        self.loader: OctreeLoader = OctreeLoader(self._octree, layer_ref)

        thumbnail_image = np.zeros(
            (32, 32, 3)
        )  # blank until we have a real one
        self.thumbnail = thumbnail_image

    @property
    def loaded(self) -> bool:
        """True if the data has been loaded.

        Because octree multiscale is async, we say we are loaded up front even
        though none of our chunks/tiles might be loaded yet.

        Returns
        -------
        bool
            True if the data as been loaded.
        """
        return self.data is not None

    @property
    def octree_level_info(self) -> Optional[OctreeLevelInfo]:
        """Information about the current octree level.

        Returns
        -------
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
                trans._(
                    "Octree level {index} is not in range(0, {num_levels})",
                    deferred=True,
                    index=index,
                    num_levels=num_levels,
                )
            ) from exc

    def get_intersection(self, view: OctreeView) -> OctreeIntersection:
        """Return the given view's intersection with the octree.

        The OctreeIntersection primarily contains the set of tiles at
        some level that need to be drawn to depict view. The "ideal level"
        is generally chosen automatically based on the screen resolution
        described by the OctreeView.

        Parameters
        ----------
        view : OctreeView
            Intersect this view with the octree.

        Returns
        -------
        OctreeIntersection
            The given view's intersection with the octree.
        """
        level = self._get_auto_level(view)
        return OctreeIntersection(level, view)

    def _get_auto_level(self, view: OctreeView) -> OctreeLevel:
        """Return the automatically selected octree level for this view.

        Parameters
        ----------
        view : OctreeView
            Get the OctreeLevel for this view.

        Returns
        -------
        OctreeLevel
            The automatically chosen OctreeLevel.
        """
        index = self._get_auto_level_index(view)
        if index < 0 or index >= self._octree.num_levels:
            raise ValueError(
                trans._(
                    "Invalid octree level {index}",
                    deferred=True,
                    index=index,
                )
            )

        return self._octree.levels[index]

    def _get_auto_level_index(self, view: OctreeView) -> int:
        """Return the automatically selected octree level index for this view.

        Parameters
        ----------
        view : OctreeView
            Get the octree level index for this view.

        Returns
        -------
        int
            The automatically chosen octree level index.
        """
        if not view.auto_level:
            # Return current level, do not update it.
            return self.octree_level

        # Find the right level automatically. Choose a level where the texels
        # in the octree tiles are around the same size as screen pixels.
        # We can do this smarter in the future, maybe have some hysteresis
        # so you don't "pop" to the next level as easily, so there is some
        # sort of dead zone between levels?
        ratio = view.data_width / view.canvas[0]

        if ratio <= 1:
            return 0  # Show the best we've got!

        # Choose the right level...
        max_level = self._octree.num_levels - 1
        return min(math.floor(math.log2(ratio)), max_level)

    def _get_octree_chunk(self, location: OctreeLocation) -> OctreeChunk:
        """Return the OctreeChunk at his location.

        Do not create the chunk if it doesn't exist.

        Parameters
        ----------
        location : OctreeLocation
            Return the chunk at this location.

        Returns
        -------
        OctreeChunk
            The returned chunk.
        """
        level = self._octree.levels[location.level_index]
        return level.get_chunk(location.row, location.col, create=False)

    def on_chunk_loaded(self, request: ChunkRequest) -> bool:
        """Called when an asynchronous ChunkRequest was loaded.

        This overrides Image.on_chunk_loaded() fully.

        Parameters
        ----------
        request : ChunkRequest
            The request for the chunk that was loaded.

        Returns
        -------
        bool
            True if the chunk's data was added to the octree.
        """
        location = request.location

        if location.slice_id != id(self):
            # There was probably a load in progress when the slice was changed.
            # The original load finished, but we are now showing a new slice.
            # Don't consider it error, just ignore the chunk.
            LOGGER.debug(
                "on_chunk_loaded: wrong slice_id: %s",
                location,
            )
            return False  # Do not add the chunk.

        octree_chunk = self._get_octree_chunk(location)

        if octree_chunk is None:
            # This location in the octree does not contain an OctreeChunk.
            # That's unexpected, because locations are turned into
            # OctreeChunk's when a load is initiated. So this is an error,
            # but log it and keep going, maybe some transient weirdness?
            LOGGER.error(
                "on_chunk_loaded: missing OctreeChunk: %s",
                octree_chunk,
            )
            return False  # Did not add the chunk.

        LOGGER.debug("on_chunk_loaded: adding %s", octree_chunk)

        # Get the data from the request.
        incoming_data = request.chunks.get('data')

        # Loaded data should always be an ndarray.
        assert isinstance(incoming_data, np.ndarray)

        # Add that data to the octree's OctreeChunk. Now the chunk can be draw.
        octree_chunk.data = incoming_data

        # Setting data should mean:
        assert octree_chunk.in_memory
        assert not octree_chunk.needs_load

        return True  # Chunk was added.
