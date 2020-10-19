"""Octree class.
"""
from typing import Tuple

import numpy as np
from scipy import ndimage as ndi

from ....types import ArrayLike


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

    def __init__(self, data: ArrayLike, pos: Tuple[float, float], size: float):
        self.data = data
        self.pos = pos
        self.size = size


def _create_tiles(array: np.ndarray, tile_size: int) -> np.ndarray:
    """
    Return an NxM array of (tile_size, tile_size) ndarrays except the edge
    tiles might be smaller if the array did not divide evenly.

    TODO_OCTREE: Maybe use slices_from_chunks from dask.array.core, but
    right now this is just for testing, real multi-scale images will
    already be chunked. Although on-the-fly chunking might be something
    we eventually tackle.
    """
    if array.ndim != 3:
        raise ValueError(f"Unexpected array dimension {array.ndim}")
    (rows, cols, _) = array.shape

    tiles = []

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


def _create_downsampled_tile(ul, ur, ll, lr) -> np.ndarray:
    """Create one tile from four child tiles.
    """
    row1 = np.hstack((ul, ur))
    row2 = np.hstack((ll, lr))
    combined_tile = np.vstack((row1, row2))

    zoom = [0.5, 0.5, 1]
    return ndi.zoom(combined_tile, zoom)


def _create_higher_level(tiles):
    """Combine each 2x2 group of tiles into one downsampled tile

    """

    new_tiles = []

    for row in range(0, len(tiles), 2):
        row_tiles = []
        for col in range(0, len(tiles[row]), 2):
            tile = _create_downsampled_tile(
                tiles[row][col],
                tiles[row][col + 1],
                tiles[row + 1][col],
                tiles[row + 1][col + 1],
            )
            row_tiles.append(tile)
        new_tiles.append(row_tiles)

    return new_tiles


def _print_level_tiles(tiles):
    """Print information about these tiles.
    """
    num_rows = len(tiles)

    num_cols = None
    for row in tiles:
        if num_cols is None:
            num_cols = len(row)
        else:
            assert num_cols == len(row)

    print(f"{num_rows} x {num_cols} = {num_rows * num_cols}")

    for row in tiles:
        for tile in row:
            pass  # print(tile.shape)


def _build_tree(levels, level_index, row, col):
    if level_index < 0:
        return None

    # print(f"Building level = {level_index}")
    level = levels[level_index]
    next_index = level_index - 1

    nrow = row * 2
    ncol = col * 2

    node = OctreeNode(row, col, level[row][col])
    node.children = [
        _build_tree(levels, next_index, nrow, ncol),
        _build_tree(levels, next_index, nrow, ncol + 1),
        _build_tree(levels, next_index, nrow + 1, ncol),
        _build_tree(levels, next_index, nrow + 1, ncol + 1),
    ]

    return node


def _print_levels(levels):
    print(f"{len(levels)} levels:")
    for level in levels:
        _print_level_tiles(level)


def _print_tiles(node, level=0):
    assert node is not None
    assert node.tile is not None
    node.print_info(level)
    for child in node.children:
        if child is not None:
            _print_tiles(child, level + 1)


class OctreeNode:
    """Octree Node.

    Child indexes
    -------------
    OCTREE_TODO: This order was picked arbitrarily, if there is another
    ordering which makes more sense, we should switch to it.

    -Z [0..3]
    +Z [4..7]

        -X X+
    -Y 0 1
    +Y 3 2

        -X X+
    -Y 4 5
    +Y 7 6
    """

    def __init__(self, row, col, tile):
        assert tile is not None
        self.row = row
        self.col = col
        self.tile = tile
        self.children = None

    def print_info(self, level):
        indent = "    " * level
        print(
            f"{indent}level={level} row={self.row:>3}, col={self.col:>3} shape={self.tile.shape}"
        )


class Octree:
    def __init__(self, root: OctreeNode, levels):
        self.root = root
        self.levels = levels  # temporary?
        self.num_levels = len(self.levels)

    def print_tiles(self):
        _print_tiles(self.root)

    @classmethod
    def from_levels(cls, levels):
        root_level = len(levels) - 1
        root = _build_tree(levels, root_level, 0, 0)
        return cls(root, levels)

    @classmethod
    def from_image(cls, image: np.ndarray):
        """Create octree from given single image.

        Parameters
        ----------
        image : ndarray
            Create the octree for this single image.
        """
        TILE_SIZE = 64
        tiles = _create_tiles(image, TILE_SIZE)
        levels = [tiles]

        # Keep combining tiles until there is one root tile.
        while len(levels[-1]) > 1:
            next_level = _create_higher_level(levels[-1])
            levels.append(next_level)

        # _print_levels(levels)

        return Octree.from_levels(levels)


if __name__ == "__main__":
    tree = Octree()
