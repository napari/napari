"""OctreeMultiscaleSlice class.

For viewing one slice of a multiscale image using an octree.
"""
from typing import Callable, List, Optional

import numpy as np

from ....components.experimental.chunk import ChunkLocation, ChunkRequest
from ....types import ArrayLike
from .._image_view import ImageView
from .octree import Octree
from .octree_intersection import OctreeIntersection
from .octree_level import OctreeLevelInfo
from .octree_util import ChunkData, ImageConfig


class OctreeMultiscaleSlice:
    """View a slice of an multiscale image using an octree."""

    def __init__(
        self,
        data,
        image_config: ImageConfig,
        image_converter: Callable[[ArrayLike], ArrayLike],
    ):
        self.data = data

        self._image_config = image_config

        self._octree = Octree.from_multiscale_data(data, image_config)
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

    def get_intersection(self, corners_2d, auto_level: bool):
        """Return the intersection with the octree."""
        if self._octree is None:
            return None
        level_index = self._get_octree_level(corners_2d, auto_level)
        level = self._octree.levels[level_index]
        return OctreeIntersection(level, corners_2d)

    def _get_octree_level(self, corners_2d, auto_level):
        if not auto_level:
            return self._octree_level

        # Find the right level automatically.
        width = corners_2d[1][1] - corners_2d[0][1]
        tile_size = self._octree.image_config.tile_size
        num_tiles = width / tile_size

        # TODO_OCTREE: compute from canvas dimensions instead
        max_tiles = 5

        # Slow way to start, redo this O(1).
        for i, level in enumerate(self._octree.levels):
            if (num_tiles / level.info.scale) < max_tiles:
                return i

        return self._octree.num_levels - 1

    def get_visible_chunks(self, corners_2d, auto_level) -> List[ChunkData]:
        """Return the chunks currently in view.

        Return
        ------
        List[ChunkData]
            The chunks inside this intersection.
        """
        intersection = self.get_intersection(corners_2d, auto_level)

        if intersection is None:
            return []

        if auto_level:
            # Set current level according to what was automatically selected.
            level_index = intersection.level.info.level_index
            self._octree_level = level_index

        # Return the chunks in this intersection.
        return intersection.get_chunks(id(self))

    def _get_chunk_data(self, location: ChunkLocation):
        level = self._octree.levels[location.level_index]
        return level.tiles[location.row][location.col]

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
            # We don't consider this an error, but it means there was a load
            # in progress when the slice was changed. So we just ignore it.
            print(f"IGNORE: wrong slice_id: {location}")
            return False  # No load.

        chunk_data = self._get_chunk_data(location)
        if not isinstance(chunk_data, ChunkData):
            # This location in the octree should have already been turned into
            # a ChunkData. When the load was initiated. So this is an unexpected
            # error, but we want to log it an keep going.
            print(f"ERROR: Octree did not have ChunkData: {chunk_data}")
            return False  # No load.

        print(f"LOADED: {chunk_data}")
        # Shove the requests's ndarray into the octree's ChunkData
        chunk_data.data = request.chunks.get('data')

        # ChunkData should no longer need to be loaded. (remove eventually)
        assert not self._get_chunk_data(location).needs_load

        # ChunkLoader should only be giving us ndarray's. (remove eventually)
        assert isinstance(self._get_chunk_data(location).data, np.ndarray)

        return True  # Chunk was loaded.
