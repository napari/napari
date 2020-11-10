"""Octree utility classes.
"""
from typing import List, Tuple

import numpy as np

from ....types import ArrayLike

TileArray = List[List[np.ndarray]]


class OctreeInfo:
    """Information about the entire octree.

    Parameters
    -----------
    base_shape : Tuple[int, int]
        The base shape of the entire image at full resolution.
    tile_size : int
        The edge length of one square tile (e.g. 256).
    """

    # TODO_OCTREE: will be namedtuple/dataclass if does not grow
    def __init__(self, base_shape: Tuple[int, int], tile_size: int):
        self.base_shape = base_shape
        self.aspect = base_shape[1] / base_shape[0]
        self.tile_size = tile_size


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
    def key(self) -> Tuple[int, int, int]:
        """The unique key for this chunk.

        Switch to __hash__? Didn't immediately work.
        """
        return (self.pos[0], self.pos[1], self.level_index)
