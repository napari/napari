"""create_multi_scale_levels() function.

This file is early/placeholder. In normal useage we might never create
tiles, because downsampling images is very slow. But for debugging and
development we do make tiles on the fly, for test images and other reasons.

Long term we might possible make tiles in the background at some point. So
as you browse a large image that doesn't have tiles, they are created in
the background. But that's pretty speculative and far out.
"""
from typing import List

import numpy as np
from scipy import ndimage as ndi

from ....utils.perf import block_timer
from .octree_util import TileArray


def _get_tile(tiles: TileArray, row, col):
    try:
        return tiles[row][col]
    except IndexError:
        return None


def _none(items):
    return all(x is None for x in items)


def _one_tile(tiles: TileArray) -> bool:
    return len(tiles) == 1 and len(tiles[0]) == 1


def create_tiles(array: np.ndarray, tile_size: int) -> np.ndarray:
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

    print(f"create_tiles array={array.shape} tile_size={tile_size}")

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
    if _none(tiles[2:4]):
        # 0 1
        # X X
        return np.hstack(tiles[0:2])
    if _none((tiles[1], tiles[3])):
        # 0 X
        # 2 X
        return np.vstack((tiles[0], tiles[2]))

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
    return ndi.zoom(
        combined_tile, [0.5, 0.5, 1], mode='nearest', prefilter=True, order=1
    )


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


def create_multi_scale_levels(image: np.ndarray, tile_size: int) -> List:
    """Turn an image into a multi-scale image with levels.

    Parameters
    ----------
    image : np.darray
        The full image to create levels from.
    tile_size : int
        The edge length for the square tiles we should use, like 256.
    """
    with block_timer("create_tiles", print_time=True):
        tiles = create_tiles(image, tile_size)

    # This is the full resolution level zero.
    levels = [tiles]

    # Create the high levels by combining tiles. With each higher level four
    # tiles from the lower level combine into one tile at the higher level.
    # All the tiles are the same size in pixels, so the four tiles are combined
    # and then sized down by half.
    #
    # We keep creating new levels by combining tiles until we create a level
    # that on has a single tile. This is the root level.

    # Keep going as long as the last level we created as more than one tile.
    while not _one_tile(levels[-1]):
        with block_timer(
            f"Create coarser level {len(levels)}:", print_time=True
        ):
            next_level = _create_coarser_level(levels[-1])
        levels.append(next_level)

    return levels


def create_multi_scale_image(
    image: np.ndarray, tile_size: int
) -> List[np.ndarray]:
    """Turn an image into a multi-scale image with levels.

    The given image is level 0, the full resolution image. Each additional
    level is downsized by half. The final root level is small enough to
    fit in one tile.

    Parameters
    ----------
    image : np.darray
        The full image to create levels from.
    """
    levels = [image]

    # Repeat until we have level that will fit in a single tile, that will
    # be come the root/highest level.
    while max(levels[-1].shape) > tile_size:
        next_level = ndi.zoom(
            levels[-1], [0.5, 0.5, 1], mode='nearest', prefilter=True, order=1
        )
        levels.append(next_level)

    return levels
