"""Octree class.
"""
import numpy as np
from scipy import ndimage as ndi

# from enum import IntEnum
# class ChildIndex(IntEnum):
#    Upper


def _create_tiles(array: np.ndarray, tile_size: int) -> np.ndarray:
    """
    Return an NxM array of (tile_size, tile_size) ndarrays except the edge
    tiles might be smaller if the array did not divide evenly.
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


def _create_tile(ul, ur, ll, lr) -> np.ndarray:
    """Create one tile from four child tiles.
    """
    row1 = np.hstack((ul, ur))
    row2 = np.hstack((ll, lr))
    full_size_tile = np.vstack((row1, row2))

    # Downsample by half.
    zoom = [0.5, 0.5, 1]
    return ndi.zoom(full_size_tile, zoom, prefilter=False, order=0)


def _combine_tiles(tiles):
    """Combine each 2x2 group of tiles into one downsampled tile

    """

    new_tiles = []

    for row in range(0, len(tiles), 2):
        row_tiles = []
        for col in range(0, len(tiles[row]), 2):
            tile = _create_tile(
                tiles[row][col],
                tiles[row][col + 1],
                tiles[row + 1][col],
                tiles[row + 1][col + 1],
            )
            row_tiles.append(tile)
        new_tiles.append(row_tiles)

    return new_tiles


def _print_tiles(tiles):
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


class Octree:
    class Node:
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

        def __init__(self):
            self.children = None

    def __init__(self):
        self.root = self.Node()

    @classmethod
    def from_image(cls, image: np.ndarray):
        TILE_SIZE = 64
        tiles = _create_tiles(image, TILE_SIZE)
        levels = [tiles]

        # Keep combining until there is one root tile.
        while len(levels[-1]) > 1:
            next_level = _combine_tiles(levels[-1])
            levels.append(next_level)

        print(f"{len(levels)} levels:")
        for level in levels:
            _print_tiles(level)


if __name__ == "__main__":
    tree = Octree()
