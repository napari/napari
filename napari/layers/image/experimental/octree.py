"""Octree class.
"""
from typing import List, Tuple

import numpy as np
from scipy import ndimage as ndi

from ....types import ArrayLike

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

    def __init__(self, data: ArrayLike, pos: Tuple[float, float], size: float):
        self.data = data
        self.pos = pos
        self.size = size


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


def _create_downsampled_tile(
    ul: np.ndarray, ur: np.ndarray, ll: np.ndarray, lr: np.ndarray
) -> np.ndarray:
    """Create one parent tile from four child tiles.

    Parameters
    ----------
    ul : np.ndarray
        Upper left child tile.
    ur : np.ndarray
        Upper right child tile.
    ll : np.ndarray
        Lower left child tile.
    lr : np.ndarray
        Lower right child tile.
    """
    row1 = np.hstack((ul, ur))
    row2 = np.hstack((ll, lr))
    combined_tile = np.vstack((row1, row2))

    # Down sample by half.
    return ndi.zoom(combined_tile, [0.5, 0.5, 1])


def _create_coarser_level(tiles: TileArray) -> TileArray:
    """Return the next coarser level of tiles.

    Combine each 2x2 group of tiles into one downsampled tile.

    tiles : TileArray
        The tiles to combine.
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


def _create_node(levels: Levels, level_index: int, row: int = 0, col: int = 0):
    """Return an Octree node and its child nodes recursively.

    Parameters
    ----------
    levels : Levels
        The tiles in each level of the octree.
    level_index : int
        Create a node at this level.
    row : int
        Create a node at this ro.
    col : int
        Create a node at this col.
    """

    if level_index < 0:
        return None

    # print(f"Building level = {level_index}")
    level = levels[level_index]
    next_index = level_index - 1

    nrow = row * 2
    ncol = col * 2

    node = OctreeNode(row, col, level[row][col])
    node.children = [
        _create_node(levels, next_index, nrow, ncol),
        _create_node(levels, next_index, nrow, ncol + 1),
        _create_node(levels, next_index, nrow + 1, ncol),
        _create_node(levels, next_index, nrow + 1, ncol + 1),
    ]

    return node


def _print_levels(levels: Levels):
    """Print information about the levels.

    Parameters
    ----------
    levels : Levels
        Print information about these levels.
    """
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
    OCTREE_TODO: This order was picked arbitrarily. If there is another
    ordering which makes more sense, we should switch to it.

    -Z [0..3]
    +Z [4..7]

        -X X+
    -Y 0 1
    +Y 3 2

        -X X+
    -Y 4 5
    +Y 7 6

    Parameters
    ----------
    row : int
        The row of this octree node in its level.
    col : int
        The col of this octree node in its level.
    data : np.ndarray
        The image data for this octree node.
    """

    def __init__(self, row: int, col: int, data: np.ndarray):
        assert data is not None
        assert row >= 0
        assert col >= 0
        self.row = row
        self.col = col
        self.data = data
        self.children = None

    def print_info(self, level):
        """Print information about this octree node.

        level : int
            The level of this node in the tree.
        """
        indent = "    " * level
        print(
            f"{indent}level={level} row={self.row:>3}, col={self.col:>3} "
            f"shape={self.tile.shape}"
        )


class Octree:
    """An octree.

    Parameters
    ----------
    root : OctreeNode
        The root of the tree.
    levels : Levels
        All the levels of the tree

    TODO_OCTREE: Do we need/want to store self.levels?
    """

    def __init__(self, root: OctreeNode, levels: Levels):
        self.root = root
        self.levels = levels
        self.num_levels = len(self.levels)

    def print_tiles(self):
        """Print information about our tiles."""
        _print_tiles(self.root)

    @classmethod
    def from_levels(cls, levels: Levels):
        """Create a tree from the given levels.

        Parameters
        ----------
        levels : Levels
            All the tiles to include in the tree.
        """
        root_level = len(levels) - 1
        root = _create_node(levels, root_level)
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
            next_level = _create_coarser_level(levels[-1])
            levels.append(next_level)

        # _print_levels(levels)

        return Octree.from_levels(levels)


if __name__ == "__main__":
    tree = Octree()
