"""OctreeImageSlice class.
"""
import logging
from typing import Callable, List

from ....types import ArrayLike
from ....utils.perf import block_timer
from .._image_slice import ImageSlice
from .octree import Octree
from .octree_intersection import OctreeIntersection
from .octree_util import ChunkData, OctreeInfo, OctreeLevelInfo

LOGGER = logging.getLogger("napari.async")


class OctreeImageSlice(ImageSlice):
    """Add Octree functionality to ImageSlice
    """

    def __init__(
        self,
        image: ArrayLike,
        image_converter: Callable[[ArrayLike], ArrayLike],
        rgb: bool,
        tile_size: int,
        octree_level: int,
    ):
        LOGGER.debug("OctreeImageSlice.__init__")
        super().__init__(image, image_converter, rgb)

        self._tile_size = tile_size
        self._octree = None
        self._octree_level = octree_level

    @property
    def num_octree_levels(self) -> int:
        """Return the number of levels in the octree.

        Return
        ------
        int
            The number of levels in the octree.
        """
        if self._octree is None:
            return 0
        else:
            return self._octree.num_levels

    def _set_raw_images(
        self, image: ArrayLike, thumbnail_source: ArrayLike
    ) -> None:
        """Set the image and its thumbnail.

        If floating point / grayscale then clip to [0..1].

        Parameters
        ----------
        image : ArrayLike
            Set this as the main image.
        thumbnail : ArrayLike
            Derive the thumbnail from this image.
        """
        super()._set_raw_images(image, thumbnail_source)

        # TODO_OCTREE: Create an octree as a test... the expection is this
        # is a *single* scale image and we create an octree on the fly just
        # so we have something to render.
        with block_timer("Octree.from_image", print_time=True):
            self._octree = Octree.from_image(image, self._tile_size)

        # Set to max level if we had no previous level (None) or if
        # our previous level was too high for this new tree.
        if (
            self._octree_level is None
            or self._octree_level >= self._octree.num_levels
        ):
            self._octree_level = 0  # self._octree.num_levels - 1

        # self._octree.print_tiles()

    def get_visible_chunks(self, corners_2d, auto_level) -> List[ChunkData]:
        """Return the chunks currently in view.

        Return
        ------
        List[ChunkData]
            The chunks inside this intersection.
        """
        intersection = self.get_intersection(corners_2d, auto_level)

        if auto_level:
            # Set current level according to what was automatically selected.
            level_index = intersection.level.info.level_index
            self._octree_level = level_index

        # Return the chunks in this intersection.
        return intersection.get_chunks()

    def get_intersection(self, corners_2d, auto_level: bool):
        """Return the intersection with the octree."""

        level_index = self._get_octree_level(corners_2d, auto_level)
        level = self._octree.levels[level_index]

        return OctreeIntersection(level, corners_2d)

    def _get_octree_level(self, corners_2d, auto_level):
        if auto_level:
            width = corners_2d[1][1] - corners_2d[0][1]
            tile_size = self._octree.info.tile_size
            num_tiles = width / tile_size

            # Compute from window width?
            MAX_TILES = 5

            # Slow way to start, redo this O(1).
            for i, level in enumerate(self._octree.levels):
                if (num_tiles / level.info.scale) < MAX_TILES:
                    return i

            return self._octree.num_levels - 1
        else:
            return self._octree_level

    @property
    def octree_info(self) -> OctreeInfo:
        if self._octree is None:
            return None
        else:
            return self._octree.info

    @property
    def octree_level_info(self) -> OctreeLevelInfo:
        if self._octree is None:
            return None
        else:
            return self._octree.levels[self._octree_level].info
