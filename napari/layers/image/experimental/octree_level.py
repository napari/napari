"""OctreeLevel and OctreeLevelInfo classes.
"""
from typing import Tuple

from .octree_util import ImageConfig, TileArray


class OctreeLevelInfo:
    """Information about one level of the octree.

    Parameters
    ----------
    image_config : ImageConfig
        Information about the entire octree.
    level_index : int
        The index of this level within the whole tree.
    shape_in_tiles : Tuple[int, int]
        The (height, width) dimensions of this level in terms of tiles.
    """

    def __init__(
        self,
        image_config: ImageConfig,
        level_index: int,
        shape_in_tiles: Tuple[int, int],
    ):
        self.image_config = image_config

        self.level_index = level_index
        self.scale = 2 ** self.level_index

        base = image_config.base_shape
        self.image_shape = (
            int(base[0] / self.scale),
            int(base[1] / self.scale),
        )

        self.shape_in_tiles = shape_in_tiles


class OctreeLevel:
    """One level of the octree.

    A level contains a 2D array of tiles.

    Soon might also contain a 3D array of sub-volumes.

    Parameters
    ----------
    image_config : ImageConfig
        Basic image configuration.
    level_index : int
        Index of this specific level (0 is full resolution).
    tile : TileArray
        The tiles for this level.
    """

    def __init__(
        self, image_config: ImageConfig, level_index: int, tiles: TileArray
    ):
        shape_in_tiles = [len(tiles), len(tiles[0])]
        self.info = OctreeLevelInfo(image_config, level_index, shape_in_tiles)
        self.tiles = tiles
