"""Functions to downsample images and create multi-scale images.

This file is early/placeholder. In normal useage we might never create
tiles, because downsampling images is very slow. But for debugging and
development we do make tiles on the fly, for test images and other reasons.

Long term we might possible make tiles in the background at some point. So
as you browse a large image that doesn't have tiles, they are created in
the background. But that's pretty speculative and far out.
"""
import time
from typing import List

import dask
import dask.array as da
import numpy as np
from scipy import ndimage as ndi

from ....types import ArrayLike
from .octree_util import NormalNoise

TileArray = List[List[ArrayLike]]


def _get_tile(tiles: TileArray, row, col):
    try:
        return tiles[row][col]
    except IndexError:
        return None


def _none(items):
    return all(x is None for x in items)


def _add_delay(array, delay_ms: NormalNoise):
    """Add a random delay when this array is first accessed.

    TODO_OCTREE: unused not but might use again...

    Parameters
    ----------
    noise : NormalNoise
        The amount of the random delay in milliseconds.
    """

    @dask.delayed
    def delayed(array):
        sleep_ms = max(0, np.random.normal(delay_ms.mean, delay_ms.std_dev))
        time.sleep(sleep_ms / 1000)
        return array

    return da.from_delayed(delayed(array), array.shape, array.dtype)


def _combine_tiles(*tiles: np.ndarray) -> np.ndarray:
    """Combine between one and four tiles into a single tile.

    The single resulting tile is not downsampled, so its size is the size
    of the four tiles combined together. However, typically the result will
    be downsampled by half in the following steps.

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
        # We only have one tile:
        # 0 X
        # X X
        return tiles[0]
    if _none(tiles[2:4]):
        # We only have the top two tiles:
        # 0 1
        # X X
        return np.hstack(tiles[0:2])
    if _none((tiles[1], tiles[3])):
        # We only have the left two tiles:
        # 0 X
        # 2 X
        return np.vstack((tiles[0], tiles[2]))

    # We have all four tiles:
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

    Combine each 2x2 group of tiles into one downsampled tile. This is slow
    so currently it's only used for testing. Most multi-scale data will
    be provided pre-downsampled into multiple levels.

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


def create_downsampled_levels(
    image: np.ndarray, tile_size: int
) -> List[np.ndarray]:
    """Return a list of levels coarser then this own.

    The first returned level is half the size of the input image, and each
    additional level is half as small again. The longest size in the
    last level is equal to or smaller than tile_size.

    For example if the tile_size is 256, the data in the file level will
    be smaller than (256, 256).

    Parameters
    ----------
    image : np.darray
        The full image to create levels from.

    Return
    ------
    List[np.ndarray]
        A list of levels where levels[0] is the first downsampled level.
    """
    zoom = [0.5, 0.5]

    if image.ndim == 3:
        zoom.append(1)  # don't downsample the colors!

    levels = []
    previous = image

    # Repeat until we have level that will fit in a single tile, that will
    # be come the root/highest level.
    while max(previous.shape) > tile_size:
        next_level = ndi.zoom(
            previous, zoom, mode='nearest', prefilter=True, order=1
        )
        levels.append(next_level)
        previous = levels[-1]

    return levels


def create_multi_scale(image: np.ndarray, tile_size: int) -> List[np.ndarray]:
    """Turn an image into a multi-scale image with levels.

    Parameters
    ----------
    image : np.darray
        The full image to create levels from.

    Return
    ------
    List[np.ndarray]
        A list of levels where levels[0] is the input image.
    """
    return [image] + create_downsampled_levels(image, tile_size)
