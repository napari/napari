"""OctreeLevel and OctreeLevelInfo classes.
"""
import math
from typing import List, Optional

import numpy as np

from ....types import ArrayLike
from .octree_chunk import OctreeChunk, OctreeChunkGeom, OctreeLocation
from .octree_util import SliceConfig


class OctreeLevelInfo:
    """Information about one level of the octree.

    Parameters
    ----------
    slice_config : SliceConfig
        Information about the entire octree.
    level_index : int
        The index of this level within the whole tree.
    shape_in_tiles : Tuple[int, int]
        The (height, width) dimensions of this level in terms of tiles.
    """

    def __init__(self, slice_config: SliceConfig, level_index: int):
        self.slice_config = slice_config

        self.level_index = level_index
        self.scale = 2 ** self.level_index

        base = slice_config.base_shape
        self.image_shape = (
            int(base[0] / self.scale),
            int(base[1] / self.scale),
        )

        tile_size = self.slice_config.tile_size
        scaled_size = tile_size * self.scale

        self.shape_in_tiles = [
            math.ceil(base[0] / scaled_size),
            math.ceil(base[1] / scaled_size),
        ]

        self.num_tiles = self.shape_in_tiles[0] * self.shape_in_tiles[1]


class OctreeLevel:
    """One level of the octree.

    A level contains a 2D array of tiles.

    Soon might also contain a 3D array of sub-volumes.

    Parameters
    ----------
    slice_id : int
        The id of the OctreeMultiscaleSlice we are in.
    data : ArrayLike
        The data for this level.
    slice_config : SliceConfig
        The base image shape and other details.
    level_index : int
        Index of this specific level (0 is full resolution).
    """

    def __init__(
        self,
        slice_id: int,
        data: ArrayLike,
        slice_config: SliceConfig,
        level_index: int,
    ):
        self.slice_id = slice_id
        self.data = data

        # TODO_OCTREE: change from "info" to "meta"/"metadata"?
        # info is kind of dumb sounding.
        self.info = OctreeLevelInfo(slice_config, level_index)
        self._tiles = {}

    def get_chunk(
        self, row: int, col: int, create=False
    ) -> Optional[OctreeChunk]:
        """Return the OctreeChunk at this location if it exists.

        If create is True, an OctreeChunk will be created if one
        does not exist at this location.

        Parameters
        ----------
        row : int
            The row in the level.
        col : int
            The column in the level.
        create : bool
            If True, create the OctreeChunk if it does not exist.

        Return
        ------
        Optional[OctreeChunk]
            The OctreeChunk if one existed or we just created it.
        """
        try:
            return self._tiles[(row, col)]
        except KeyError:
            if not create:
                return None  # It didn't exist so we're done.

        # Create a chunk at this location and return it.
        octree_chunk = self._create_chunk(row, col)
        self._tiles[(row, col)] = octree_chunk
        return octree_chunk

    def _create_chunk(self, row: int, col: int) -> OctreeChunk:
        """Create a new OctreeChunk for this location in the level.

        Parameters
        ----------
        row : int
            The row in the level.
        col : int
            The column in the level.

        Return
        ------
        OctreeChunk
            The newly created chunk.
        """
        level_index = self.info.level_index
        location = OctreeLocation(self.slice_id, level_index, row, col)

        scale = self.info.scale
        scale_vec = np.array([scale, scale], dtype=np.float32)

        tile_size = self.info.slice_config.tile_size
        scaled_size = tile_size * scale

        pos = np.array(
            [col * scaled_size, row * scaled_size], dtype=np.float32
        )

        # Geom is used by the visual for rendering this chunk.
        geom = OctreeChunkGeom(pos, scale_vec)

        data = self._get_data(row, col)

        # Return the newly created chunk.
        return OctreeChunk(data, location, geom)

    def _get_data(self, row: int, col: int) -> ArrayLike:

        tile_size = self.info.slice_config.tile_size

        array_slice = (
            slice(row * tile_size, (row + 1) * tile_size),
            slice(col * tile_size, (col + 1) * tile_size),
        )

        if self.data.ndim == 3:
            array_slice += (slice(None),)  # Add the colors.

        data = self.data[array_slice]

        return data


def print_levels(
    label: str, levels: List[OctreeLevel], start: int = 0
) -> None:
    """Print the dimensions of each level nicely.

    Parameters
    ----------
    label : str
        Prepend this to the header line.
    levels : List[OctreeLevel]
        Print information about these levels.
    start : int
        Start the indexing at this number, shift the indexes up.
    """
    from ...._vendor.experimental.humanize.src.humanize import intword

    def _dim_str(dim: tuple) -> None:
        return f"{dim[0]} x {dim[1]} = {intword(dim[0] * dim[1])}"

    print(f"{label} {len(levels)} levels:")
    for index, level in enumerate(levels):
        level_index = start + index
        image_str = _dim_str(level.info.image_shape)
        tiles_str = _dim_str(level.info.shape_in_tiles)
        print(
            f"    Level {level_index}: {image_str} pixels -> {tiles_str} tiles"
        )
