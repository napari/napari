"""Octree class.
"""
from typing import List, Tuple

import numpy as np
from scipy import ndimage as ndi

from ....types import ArrayLike
from ....utils.perf import block_timer

TileArray = List[List[np.ndarray]]
Levels = List[TileArray]


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


def _create_tiles(array: np.ndarray, tile_size: int) -> np.ndarray:
    """
    Return an NxM array of (tile_size, tile_size) ndarrays except the edge
    tiles might be smaller if the array did not divide evenly.

    TODO_OCTREE: slices_from_chunks from dask.array.core possibly does
    the same thing, if we are going to use this in production.

    Parameters
    ----------
    array : np.ndarray
        The array to create tiles out of.
    tiles_size : int
        Edge length of the square tiles.
    """
    if array.ndim != 3:
        raise ValueError(f"Unexpected array dimension {array.ndim}")
    (rows, cols, _) = array.shape

    tiles = []

    print(f"_create_tiles array={array.shape} tile_size={tile_size}")

    row = 0
    while row < rows:
        row_tiles = []
        col = 0
        while col < cols:
            tile = array[row : row + tile_size, col : col + tile_size, :]
            row_tiles.append(tile)
            col += tile_size
        tiles.append(row_tiles)
        row += tile_size

    return tiles


def _get_tile(tiles: TileArray, row, col):
    try:
        return tiles[row][col]
    except IndexError:
        return None


def _none(items):
    return all(x is None for x in items)


def _combine_tiles(*tiles: np.ndarray) -> np.ndarray:
    """Combine 1-4 tiles into a single tile.

    Parameters
    ----------
    tiles
        The 4 child tiles, some might be None.
    """
    if len(tiles) != 4:
        raise ValueError("Must have 4 values")

    if tiles[0] is None:
        raise ValueError("Position 0 cannot be None")

    # The layout of the children is:
    # 0 1
    # 2 3
    if _none(tiles[1:4]):
        # 0 X
        # X X
        return tiles[0]
    elif _none(tiles[2:4]):
        # 0 1
        # X X
        return np.hstack(tiles[0:2])
    elif _none((tiles[1], tiles[3])):
        # 0 X
        # 2 X
        return np.vstack((tiles[0], tiles[2]))
    else:
        # 0 1
        # 2 3
        row1 = np.hstack(tiles[0:2])
        row2 = np.hstack(tiles[2:4])
        return np.vstack((row1, row2))


def _create_downsampled_tile(*tiles: np.ndarray) -> np.ndarray:
    """Create one parent tile from four child tiles.

    Parameters
    ----------
    tiles
        The 4 child tiles, some could be None.
    """
    # Combine 1-4 tiles together.
    combined_tile = _combine_tiles(*tiles)

    # Down sample by half.
    return ndi.zoom(combined_tile, [0.5, 0.5, 1])


def _create_coarser_level(tiles: TileArray) -> TileArray:
    """Return a level that is one level coarser.

    Combine each 2x2 group of tiles into one downsampled tile.

    Parameters
    ----------
    tiles : TileArray
        The tiles to combine.

    Returns
    -------
    TileArray
        The coarser level of tiles.
    """

    level = []

    for row in range(0, len(tiles), 2):
        row_tiles = []
        for col in range(0, len(tiles[row]), 2):
            # The layout of the children is:
            # 0 1
            # 2 3
            group = (
                _get_tile(tiles, row, col),
                _get_tile(tiles, row, col + 1),
                _get_tile(tiles, row + 1, col),
                _get_tile(tiles, row + 1, col + 1),
            )
            tile = _create_downsampled_tile(*group)
            row_tiles.append(tile)
        level.append(row_tiles)

    return level


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


class OctreeLevel:
    """One level of the octree.

    A level contains a 2D or 3D array of tiles.
    """

    def __init__(self, info: OctreeInfo, level_index: int, tiles: TileArray):
        self.info = info
        self.level_index = level_index
        self.tiles = tiles

        self.scale = 2 ** self.level_index
        self.num_rows = len(self.tiles)
        self.num_cols = len(self.tiles[0])

    def print_info(self):
        """Print information about this level."""
        nrows = len(self.tiles)
        ncols = len(self.tiles[0])
        print(f"level={self.level_index} dim={nrows}x{ncols}")

    def get_intersection(self, data_corners) -> OctreeIntersection:
        # TODO_OCTREE: generalize with data_corner indices we need to use.
        data_rows = [data_corners[0][1], data_corners[1][1]]
        data_cols = [data_corners[0][2], data_corners[1][2]]

        row_range = self.row_range(data_rows)
        col_range = self.column_range(data_cols)

        return OctreeIntersection(
            [self.num_rows, self.num_cols], [row_range, col_range]
        )

    def get_chunks(self, data_corners) -> List[ChunkData]:
        """Return chunks that are within this rectangular region of the data.

        Parameters
        ----------
        data_corners
            Return chunks within this rectangular region.
        """
        chunks = []

        intersection = self.get_intersection(data_corners)

        scale = self.scale
        scale_vec = [scale, scale]

        tile_size = self.info.tile_size

        ranges = intersection.ranges

        # Iterate over every tile in the rectangular region.
        data = None
        y = ranges[0].start * tile_size
        for row in ranges[0]:
            x = ranges[1].start * tile_size
            for col in ranges[1]:

                data = self.tiles[row][col]
                pos = [x, y]

                if 0 not in data.shape:
                    chunks.append(ChunkData(data, pos, scale_vec))

                x += data.shape[1] * scale
            y += data.shape[0] * scale

        return chunks

    def tile_range(self, span, num_tiles):
        """Return tiles indices needed to draw the span."""

        def _clamp(val, min_val, max_val):
            return max(min(val, max_val), min_val)

        tile_size = self.info.tile_size

        tiles = [span[0] / tile_size, span[1] / tile_size]
        new_min = _clamp(tiles[0], 0, num_tiles - 1)
        new_max = _clamp(tiles[1], 0, num_tiles - 1)
        clamped = [new_min, new_max + 1]

        span_int = [int(x) for x in clamped]
        return range(*span_int)

    def row_range(self, span):
        """Return row indices which span image coordinates [y0..y1]."""
        return self.tile_range(span, self.num_rows)

    def column_range(self, span):
        """Return column indices which span image coordinates [x0..x1]."""
        return self.tile_range(span, self.num_cols)


def _one_tile(tiles: TileArray) -> bool:
    return len(tiles) == 1 and len(tiles[0]) == 1


class Octree:
    """A region octree that holds hold 2D or 3D images.

    Today the octree is full/complete meaning every node has 4 or 8
    children, and every leaf node is at the same level of the tree. This
    makes sense for region/image trees, because the image exists
    everywhere.

    Since we are a complete tree we don't need actual nodes with references
    to the node's children. Instead, every level is just an array, and
    going from parent to child or child to parent is trivial, you just
    need to double or half the indexes.

    Future Work: Geometry
    ---------------------
    Eventually we want our octree to hold geometry, not just images.
    Geometry such as points and meshes. For geometry a sparse octree might
    make more sense than this full/complete region octree.

    With geometry there might be lots of empty space in between small dense
    pockets of geometry. Some parts of tree might need to be very deep, but
    it would be a waste for the tree to be that deep everywhere.

    Parameters
    ----------
    base_shape : Tuple[int, int]
        The shape of the full base image.
    levels : Levels
        All the levels of the tree.
    """

    def __init__(self, base_shape: Tuple[int, int], levels: Levels):
        self.base_shape = base_shape
        self.levels = [
            OctreeLevel(base_shape, i, level)
            for (i, level) in enumerate(levels)
        ]
        self.num_levels = len(self.levels)

    def print_info(self):
        """Print information about our tiles."""
        for level in self.levels:
            level.print_info()

    @classmethod
    def from_image(cls, image: np.ndarray, tile_size: int):
        """Create octree from given single image.

        Parameters
        ----------
        image : ndarray
            Create the octree for this single image.
        """
        info = OctreeInfo(image.shape, tile_size)

        with block_timer("create_tiles", print_time=True):
            tiles = _create_tiles(image, info.tile_size)
        levels = [tiles]

        # Keep combining tiles until there is one root tile.
        while not _one_tile(levels[-1]):
            with block_timer(
                f"Create coarser level {len(levels)}:", print_time=True
            ):
                next_level = _create_coarser_level(levels[-1])
            levels.append(next_level)

        # _print_levels(levels)

        return Octree(info, levels)


if __name__ == "__main__":
    tree = Octree()
