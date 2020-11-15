"""OctreeMultiscaleSlice class.

For viewing one slice of a multiscale image using an octree.
"""
from typing import Callable, List, Optional

import numpy as np

from ....types import ArrayLike
from .._image_slice_data import ImageSliceData
from .._image_view import ImageView
from .octree import Octree
from .octree_intersection import OctreeIntersection
from .octree_level import OctreeLevelInfo
from .octree_util import ChunkData


class OctreeMultiscaleSlice:
    """View a slice of an multiscale image using an octree."""

    def __init__(
        self,
        tile_size: int,
        image_converter: Callable[[ArrayLike], ArrayLike],
    ):
        self._tile_size = tile_size

        # OCTREE_TODO: None until loaded for now, but can we change it so
        # the data is loaded as soon as OctreeMultiscaleSlice is created?
        # Better to have loaded and unloaded states.
        self.data = None
        self._octree_level = None
        self._octree = None

        thumbnail_image = np.zeros(
            (64, 64, 3)
        )  # blank until we have a real one
        self.thumbnail: ImageView = ImageView(thumbnail_image, image_converter)

    @property
    def octree_level(self) -> int:
        # TODO_OCTREE: just have an exposed int instead?
        return self._octree_level

    @property
    def loaded(self) -> bool:
        """Return True if the data has been loaded.

        Because octree multiscale is async, we say we are loaded up front even
        though none of our chunks/tiles might be loaded yet.
        """
        return self.data is not None

    def load(self, data: ImageSliceData) -> bool:
        """Load this data into the slice.

        Parameters
        ----------
        data : ImageSliceData
            The data to load into this slice.

        Return
        ------
        bool
            Return True if load was synchronous.
        """
        self.data = data
        self._octree = Octree.from_multiscale_data(data, self._tile_size)

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
        tile_size = self._octree.info.tile_size
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
        return intersection.get_chunks()
