"""Octree class.
"""
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from ....types import ArrayLike

TileArray = List[List[np.ndarray]]
Levels = List[TileArray]

TILE_SIZE = 64


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
        size: Tuple[float, float],
    ):
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


def _combine_tiles(*tiles) -> np.ndarray:
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


def _create_downsampled_tile(
    tiles: Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]
) -> np.ndarray:
    """Create one parent tile from four child tiles.

    Parameters
    ----------
    tiles
        The 4 child tiles, some could be None.
    """
    # Combine 1-4 tiles together.
    combined_tile = _combine_tiles(tiles)

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
            # The layout of the children is:
            # 0 1
            # 2 3
            tile = _create_downsampled_tile(
                (
                    _get_tile(tiles, row, col),
                    _get_tile(tiles, row, col + 1),
                    _get_tile(tiles, row + 1, col),
                    _get_tile(tiles, row + 1, col + 1),
                )
            )
            row_tiles.append(tile)
        new_tiles.append(row_tiles)

    return new_tiles


class OctreeLevel:
    """One level of the octree.

    A level contains a 2D or 3D array of tiles.
    """

    def __init__(self, image_shape, level_index: int, tiles: TileArray):
        self.image_shape = image_shape
        self.level_index = level_index
        self.tiles = tiles
        self.num_rows = len(self.tiles)
        self.num_cols = len(self.tiles[0])

    def print_info(self):
        """Print information about this level."""
        nrows = len(self.tiles)
        ncols = len(self.tiles[0])
        print(f"level={self.level_index} dim={nrows}x{ncols}")

    def get_chunks(self, data_corners) -> List[ChunkData]:
        """Return chunks that are within this rectangular region of the data.

        Parameters
        ----------
        data_corners
            Return chunks within this rectangular region.
        """
        chunks = []

        # TODO_OCTREE: generalize with data_corner indices we need to use.
        data_rows = [data_corners[0][1], data_corners[1][1]]
        data_cols = [data_corners[0][2], data_corners[1][2]]

        print(f"get_chunks rows={data_rows} cols={data_cols}")

        # Iterate over every tile in the rectangular region.
        for row in self.row_range(data_rows):
            for col in self.column_range(data_cols):

                data = self.tiles[row][col]

                # The [X, Y] position of this tile in from [0..1] range
                # where 1 is the size of the full image.
                pos_normalized = np.array(
                    (col / self.num_cols, row / self.num_rows)
                )
                pos = pos_normalized * self.image_shape[:2]

                # The [X, Y] shape of this specific tile, if it's an edge or
                # corner it might be smaller than full size.
                tile_shape = np.array([data.shape[1], data.shape[0]])

                # The [X, Y] fractional size of the tile, 1.0 means it's
                # a full size tile. Edge and corners can be smaller.
                tile_fraction = tile_shape / TILE_SIZE

                # The [X, Y] size of the tile relative to the full image,
                # where 1.0 means it spans the full image.
                tile_size = tile_fraction / (self.num_cols, self.num_rows)

                print(f"ChunkData pos={pos} size={tile_size}")
                chunks.append(ChunkData(data, pos, tile_size))

        return chunks

    def tile_range(self, span, num_tiles):
        """Return tiles indices for image coordinates [span[0]..span[1]]."""
        tile_span = [span[0] / TILE_SIZE, (span[1] / TILE_SIZE) + 1]
        tile_span = [max(tile_span[0], 0), min(tile_span[1], num_tiles)]
        return range(int(tile_span[0]), int(tile_span[1]))

    def row_range(self, span):
        """Return row indices which span image coordinates [y0..y1]."""
        return self.tile_range(span, self.num_rows)

    def column_range(self, span):
        """Return column indices which span image coordinates [x0..x1]."""
        return self.tile_range(span, self.num_cols)


class Octree:
    """A region octree to hold 2D or 3D images.

    Today the octree is full/complete meaning every node has 4 or 8
    children, and every leaf node is at the same level of the tree. This
    makes sense for region/image trees, because the image exists
    everywhere.

    Since we are a complete tree we don't need actual nodes with references
    to the node's children. Instead, every level is just an array, and
    going from parent to child or child to parent is trivial, you just
    need to double or half the indexes.


    Future Work
    -----------
    Support geometry, like points and meshes, not just images. For geometry
    a sparse octree might make more sense. With geometry there might be
    lots of empty space in between small dense pockets of geometry. Some
    parts of tree might need to be very deep, but it would be a waste to be
    that deep everywhere.

    Parameters
    ----------
    levels : Levels
        All the levels of the tree
    """

    def __init__(self, image_shape, levels: Levels):
        self.image_shape = image_shape
        self.levels = [
            OctreeLevel(image_shape, i, level)
            for (i, level) in enumerate(levels)
        ]
        self.num_levels = len(self.levels)

    def print_info(self):
        """Print information about our tiles."""
        for level in self.levels:
            level.print_info()

    @classmethod
    def from_image(cls, image: np.ndarray):
        """Create octree from given single image.

        Parameters
        ----------
        image : ndarray
            Create the octree for this single image.
        """
        image_shape = image.shape
        tiles = _create_tiles(image, TILE_SIZE)
        levels = [tiles]

        # Keep combining tiles until there is one root tile.
        while len(levels[-1]) > 1:
            next_level = _create_coarser_level(levels[-1])
            levels.append(next_level)

        # _print_levels(levels)

        return Octree(image_shape, levels)


if __name__ == "__main__":
    tree = Octree()
