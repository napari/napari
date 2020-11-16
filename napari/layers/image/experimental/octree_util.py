"""Octree utility classes.
"""
from typing import List, NamedTuple, Tuple

import numpy as np

from ....components.experimental.chunk import ChunkLocation
from ....types import ArrayLike

TileArray = List[List[np.ndarray]]


class ImageConfig(NamedTuple):
    """Configuration for a tiled image."""

    base_shape: Tuple[int, int]
    aspect: float
    tile_size: int
    delay_ms: float  # For testing, add a delay to tile access.

    @classmethod
    def create(
        cls,
        base_shape: Tuple[int, int],
        tile_size: int,
        delay_ms: float = None,
    ):
        """Create ImageConfig."""
        aspect = base_shape[1] / base_shape[0]
        return cls(base_shape, aspect, tile_size, delay_ms)


class ChunkData:
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

    def __init__(self, data: ArrayLike, location: ChunkLocation):
        self._data = data
        self.location = location
        self.loading = False

    def __str__(self):
        return f"{self.location}"

    @property
    def data(self) -> ArrayLike:
        return self._data

    @data.setter
    def data(self, data: np.ndarray) -> None:
        try:
            assert not self.in_memory  # Should not set twice.
        except AssertionError:
            pass
        print(f"set_data {self}")
        self._data = data
        self.loading = False

    @property
    def key(self) -> Tuple[int, int, int]:
        """The unique key for this chunk.

        Switch to __hash__? Didn't immediately work.
        """
        return (
            self.location.pos[0],
            self.location.pos[1],
            self.location.level_index,
        )

    @property
    def in_memory(self) -> bool:
        """Return True if the data is fully in memory."""
        return isinstance(self.data, np.ndarray)

    @property
    def needs_load(self) -> bool:
        return not self.in_memory and not self.loading
