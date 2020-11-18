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
    """Configuration for a tiled image."""

    base_shape: Tuple[int, int]
    num_levels: int
    tile_size: int
    delay_ms: NormalNoise = NormalNoise()

    @property
    def aspect_ratio(self):
        """Return the width:height aspect ratio of the base image.

        For example HDTV resolution is 16:9 which is 1.77.
        """
        return self.base_shape[1] / self.base_shape[0]
