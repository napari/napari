"""create_downsampled_levels()
"""
import logging
import time
from typing import List

import dask
import dask.array as da
import numpy as np
from scipy import ndimage as ndi

from napari.utils.perf import block_timer

from .octree_util import NormalNoise

LOGGER = logging.getLogger("napari.octree")


def add_delay(array, delay_ms: NormalNoise):
    """Add a random delay when this array is first accessed.

    TODO_OCTREE: unused not but might use again...

    Parameters
    ----------
    delay_ms : NormalNoise
        The amount of the random delay in milliseconds.
    """

    @dask.delayed
    def delayed(array):
        sleep_ms = max(0, np.random.normal(delay_ms.mean, delay_ms.std_dev))
        time.sleep(sleep_ms / 1000)
        return array

    return da.from_delayed(delayed(array), array.shape, array.dtype)


def create_downsampled_levels(
    image: np.ndarray, next_level_index: int, tile_size: int
) -> List[np.ndarray]:
    """Return a list of levels coarser then this own.

    The first returned level is half the size of the input image, and each
    additional level is half as small again. The longest size in the
    last level is equal to or smaller than tile_size.

    For example if the tile_size is 256, the data in the file level will
    be smaller than (256, 256).

    Notes
    -----
    Currently we use create_downsampled_levels() from Octree._create_extra_levels
    so that the image pyramid extends up to the point where the coarsest level
    fits within a single tile.

    This is potentially quite slow and wasteful. A better long term solution might
    be if our tiled visuals supported larger tiles, and a mix of tile sizes. Then
    the root level could be a special case that had a larger tiles size than
    the interior levels. This would mean zero downsampled, it'd probably perform
    better. Tiling an image that the graphics card can easily display is
    probably not efficient.

    Parameters
    ----------
    image : np.ndarray
        The full image to create levels from.

    Returns
    -------
    List[np.ndarray]
        A list of levels where levels[0] is the first downsampled level.
    """
    zoom = [0.5, 0.5]

    if image.ndim == 3:
        zoom.append(1)  # don't downsample the colors!

    # ndi.zoom doesn't support float16, so convert to float32
    if image.dtype == np.float16:
        image = image.astype(np.float32)

    levels = []
    previous = image
    level_index = next_level_index

    if max(previous.shape) > tile_size:
        LOGGER.info("Downsampling levels to a single tile...")

    # Repeat until we have level that will fit in a single tile, that will
    # be come the root/highest level.
    while max(previous.shape) > tile_size:
        with block_timer("downsampling") as timer:
            next_level = ndi.zoom(
                previous, zoom, mode='nearest', prefilter=True, order=1
            )

        LOGGER.info(
            "Level %d downsampled %s in %.3fms",
            level_index,
            previous.shape,
            timer.duration_ms,
        )

        levels.append(next_level)
        previous = levels[-1]
        level_index += 1

    return levels
