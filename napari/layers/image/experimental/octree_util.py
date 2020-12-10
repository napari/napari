"""Octree utility classes.
"""
from dataclasses import dataclass
from typing import NamedTuple, Tuple

import numpy as np

from ....utils.config import octree_config


def _get_tile_size() -> int:
    """Return the default tile size.

    Return
    ------
    int
        The default tile size.
    """
    return octree_config['octree']['tile_size'] if octree_config else 256


@dataclass
class OctreeDisplayOptions:
    """Options for how to display the octree.

    Attributes
    -----------
    tile_size : int
        The size of the display tiles, for example 256.
    freeze_level : bool
        If True we do not automatically pick the right data level.
    track_view : bool
        If True the displayed tiles track the view, the normal mode.
    show_grid : bool
        If True draw a grid around the tiles for debugging or demos.
    """

    def __init__(self):
        self._show_grid = True

        # TODO_OCTREE we set this after __init__ which is messy.
        self.loaded_event = None

    @property
    def show_grid(self) -> bool:
        """True if we are drawing a grid on top of the tiles.

        Return
        ------
        bool
            True if we are drawing a grid on top of the tiles.
        """
        return self._show_grid

    @show_grid.setter
    def show_grid(self, show: bool) -> None:
        """Set whether we should draw a grid on top of the tiles.

        Parameters
        ----------
        show : bool
            True if we should draw a grid on top of the tiles.
        """
        if self._show_grid != show:
            self._show_grid = show
            self.loaded_event()  # redraw

    tile_size: int = _get_tile_size()
    freeze_level: bool = False
    track_view: bool = True


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

    @property
    def aspect_ratio(self):
        """Return the width:height aspect ratio of the base image.

        For example HDTV resolution is 16:9 which is 1.77.
        """
        return self.base_shape[1] / self.base_shape[0]
