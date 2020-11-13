"""Octree utility classes.
"""
from typing import List, NamedTuple, Tuple

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


class ChunkData(NamedTuple):
    """One chunk of the full image.

    A chunk is a 2D tile or a 3D sub-volume.

    We include level_index because id(data) is sometimes duplicated in #
    adjacent levels, somehow. But it makes sense to include it anyway,
    it's an important aspect of the chunk.

    Attributes
    ----------
    level_index : int
        The octree level where this chunk is from.
    data : ArrayLike
        The data to draw for this chunk.
    pos : np.ndarray
        The x, y coordinates of the chunk.
    scale : np.ndarray
        The (x, y) scale of this chunk. Should be square/cubic.
    """

    level_index: int
    data: ArrayLike
    pos: np.ndarray
    scale: np.ndarray

    @property
    def key(self) -> Tuple[int, int, int]:
        """The unique key for this chunk.

        Switch to __hash__? Didn't immediately work.
        """
        return (self.pos[0], self.pos[1], self.level_index)
