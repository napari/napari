"""Octree utility classes.
"""
from typing import List, Tuple

import numpy as np

from ....types import ArrayLike

TileArray = List[List[np.ndarray]]


class OctreeInfo:
    def __init__(self, base_shape, tile_size: int):
        self.base_shape = base_shape
        self.tile_size = tile_size


class OctreeIntersection:
    def __init__(self, shape: Tuple[int, int], ranges: Tuple[range, range]):
        self.shape = shape
        self.ranges = ranges

    def is_visible(self, row, col):
        def _inside(value, value_range):
            return value >= value_range.start and value < value_range.stop

        return _inside(row, self.ranges[0]) and _inside(col, self.ranges[1])


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
        data: ArrayLike,
        pos: Tuple[float, float],
        scale: Tuple[float, float],
    ):
        self.data = data
        self.pos = pos
        self.scale = scale
