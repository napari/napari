"""Octree utility classes.
"""
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from ....components.experimental.chunk import ChunkKey
from ....layers import Layer
from ....types import ArrayLike

TileArray = List[List[np.ndarray]]


class TestImageSettings(NamedTuple):
    """Settings for a test image we are creating."""

    base_shape: Tuple[int, int]
    tile_size: int


class NormalNoise(NamedTuple):
    mean: float = 0
    std_dev: float = 0

    @property
    def is_zero(self) -> bool:
        """Return true if there is no noise at all."""
        return self.mean == 0 and self.std_dev == 0


class OctreeChunkGeom(NamedTuple):
    """Position and scale of the chunk, for rendering."""

    pos: np.ndarray
    scale: np.ndarray


class OctreeLocation(NamedTuple):
    """Location of one chunk within the octree."""

    slice_id: int
    level_index: int
    row: int
    col: int

    def __str__(self):
        return (
            f"location=({self.level_index}, {self.row}, {self.col}) "
            f"slice={self.slice_id} id={id(self)}"
        )

    @classmethod
    def create_null(cls):
        """Create null location that points to nothing."""
        return cls(0, 0, 0, 0, np.zeros(0), np.zeros(0))


class OctreeChunkKey(ChunkKey):
    """Add octree specific identity information to the generic ChunkKey.

    Parameters
    ----------
    layer : Layer
        The OctreeImage layer.
    indices : Tuple[Optional[slice], ...]
        The indices of the image we are viewing.
    location : OctreeLocation
        The location of the chunk within the octree we are loading.
    """

    def __init__(
        self,
        layer: Layer,
        indices: Tuple[Optional[slice], ...],
        location: OctreeLocation,
    ):
        self.location = location
        super().__init__(layer, indices)

    def _get_hash_values(self):
        # TODO_OCTREE: can't we just has with parent's hashed key instead
        # of creating a single big has value? Probably.
        parent = super()._get_hash_values()
        return parent + (self.location,)


class ImageConfig(NamedTuple):
    """Configuration for a tiled image."""

    base_shape: Tuple[int, int]
    num_levels: int
    tile_size: int
    delay_ms: NormalNoise = NormalNoise()

    @property
    def aspect_ratio(self):
        """Return the width:height aspect ratio of the base image.

        For example HDTV resolution is 16:9 which is 1.77.
        """
        return self.base_shape[1] / self.base_shape[0]


class OctreeChunk:
    """One chunk of the full 2D or 3D image in the octree.

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

    def __init__(
        self, data: ArrayLike, location: OctreeLocation, geom: OctreeChunkGeom
    ):
        self._data = data
        self._orig_data = data  # For now hold on to implement clear()
        self.location = location
        self.geom = geom
        self.loading = False

    def __str__(self):
        return f"{self.location}"

    @property
    def data(self) -> ArrayLike:
        """Return the data associated with this chunk."""
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
            self.geom.pos[0],
            self.geom.pos[1],
            self.location.level_index,
        )

    @property
    def in_memory(self) -> bool:
        """Return True if the data is fully in memory."""
        return isinstance(self.data, np.ndarray)

    @property
    def needs_load(self) -> bool:
        """Return true if this chunk needs to loaded.

        An unloaded chunk's data might be a Dask or similar deferred array.
        A loaded chunk's data is always ndarray, It's always real binary
        data in memory.
        """
        return not self.in_memory and not self.loading

    def clear(self) -> None:
        """Clear out our loaded data, return to the original.

        This is only done when running without the cache, so that we reload
        the data again. With computation the loaded data might be different
        each time.
        """
        self._data = self._orig_data
        self.loading = False
