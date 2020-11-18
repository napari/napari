"""Octree utility classes.
"""
from typing import NamedTuple, Tuple

import numpy as np


class TestImageSettings(NamedTuple):
    """Settings for a test image we are creating."""

    base_shape: Tuple[int, int]
    tile_size: int


class NormalNoise(NamedTuple):
    """Noise with a normal distribution."""

    mean: float = 0
    std_dev: float = 0

    @property
    def is_zero(self) -> bool:
        """Return True if there is no noise at all.

        Return
        ------
        bool
            True if there is no noise at all.
        """
        return self.mean == 0 and self.std_dev == 0

    @property
    def get_value(self) -> float:
        """Get a random value.

        Return
        ------
        float
            The random value.
        """
        return np.random.normal(self.mean, self.std_dev)


class SliceConfig(NamedTuple):
    """Configuration for a tiled image.

    Attributes
    ----------
    base_shape : np.ndarray
        The base [height, width] shape of the entire full resolution image.
    num_levels : int
        The number of octree levels in the image.
    tile_size : int
        The default tile size. However each OctreeLevel has its own tile size
        which can override this.

    Notes
    -----
    This SliceConfig.tile_size will be used by the OctreeLevels in the tree
    in general. But the highest level OctreeLevel might use a larger size
    so that it can consist of a single chunk.

    For example we might be using 256x256 tiles in general. For best
    performance it might make sense to have octree levels such that the
    highest level fits inside a single 256x256 tiles.

    But if we are displaying user provided data, they did not know our tile
    size. Instead their root level might be something pretty big, like
    6000x6000. In that case we use 6000x6000 as the tile size in our root,
    so the root level consists of a single tile.

    TODO_OCTREE: we don't actually support larger size tiles yet! However
    it's still a good idea to assume that each OctreeLevel could have its
    own tile size.
    """

    base_shape: np.ndarray
    num_levels: int
    tile_size: int
    delay_ms: NormalNoise = NormalNoise()

    @property
    def aspect_ratio(self):
        """Return the width:height aspect ratio of the base image.

        For example HDTV resolution is 16:9 which is 1.77.
        """
        return self.base_shape[1] / self.base_shape[0]
