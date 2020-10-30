"""Octree utility classes.
"""
from typing import List, Tuple

import numpy as np

from ....types import ArrayLike

TileArray = List[List[np.ndarray]]

# TODO_OCTREE: These types might be a horrible idea but trying it for now.
Int2 = np.ndarray  # [x, x] dtype=numpy.int32


# TODO_OCTREE: this class is placeholder, needs work
class OctreeInfo:
    def __init__(self, base_shape, tile_size: int):
        self.base_shape = base_shape
        self.aspect = base_shape[1] / base_shape[0]
        self.tile_size = tile_size


class OctreeLevelInfo:
    def __init__(
        self, octree_info: OctreeInfo, level_index: int, tile_shape: Int2
    ):
        self.octree_info = octree_info

        self.level_index = level_index
        self.scale = 2 ** self.level_index

        base = self.octree_info.base_shape
        self.image_shape = (
            int(base[0] / self.scale),
            int(base[1] / self.scale),
        )

        self.tile_shape = tile_shape


# TODO_OCTREE: this class is placeholder, needs work
class ChunkData:
    """One chunk of the full image.

    A chunk is a 2D tile or a 3D sub-volume.

    Parameters
    ----------
    data : ArrayLike
        The data to draw for this chunk.
    pos : Tuple[float, float]
        The x, y coordinates of the chunk.
    size : float
        The size of the chunk, the chunk is square/cubic.
    """

    def __init__(
        self,
        level_index: int,
        data: ArrayLike,
        pos: Tuple[float, float],
        scale: Tuple[float, float],
    ):
        # We need level_index because id(data) is sometimes duplicated in
        # adjacent layers, somehow. But it makes sense to include it
        # anyway, it's an important aspect of the chunk.
        self.level_index = level_index
        self.data = data
        self.pos = pos
        self.scale = scale

    @property
    def key(self):
        return (self.pos[0], self.pos[1], self.level_index)
