"""OctreeDisplayOptions, NormalNoise and OctreeMetadata classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from napari.utils.config import octree_config

if TYPE_CHECKING:
    from napari.components.experimental.chunk import LayerRef


def _get_tile_size() -> int:
    """Return the default tile size.

    Returns
    -------
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

    def __init__(self) -> None:
        self._show_grid = True

        # TODO_OCTREE we set this after __init__ which is messy.
        self.loaded_event = None

    @property
    def show_grid(self) -> bool:
        """True if we are drawing a grid on top of the tiles.

        Returns
        -------
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


class NormalNoise(NamedTuple):
    """Noise with a normal distribution."""

    mean: float = 0
    std_dev: float = 0

    @property
    def is_zero(self) -> bool:
        """Return True if there is no noise at all.

        Returns
        -------
        bool
            True if there is no noise at all.
        """
        return self.mean == 0 and self.std_dev == 0

    @property
    def get_value(self) -> float:
        """Get a random value.

        Returns
        -------
        float
            The random value.
        """
        return np.random.normal(self.mean, self.std_dev)


class OctreeMetadata(NamedTuple):
    """Metadata for an Octree.

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
    This OctreeMetadata.tile_size will be used by the OctreeLevels in the tree
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

    layer_ref: LayerRef
    base_shape: np.ndarray
    num_levels: int
    tile_size: int

    @property
    def aspect_ratio(self):
        """Return the width:height aspect ratio of the base image.

        For example HDTV resolution is 16:9 which has aspect ration 1.77.
        """
        return self.base_shape[1] / self.base_shape[0]


def spiral_index(row_range, col_range):
    """Generate a spiral index from a set of row and column indices.

    A spiral index starts at the center point and moves out in a spiral
    Paramters
    ---------
    row_range : range
        Range of rows to be accessed.
    col_range : range
        Range of columns to be accessed.

    Returns
    -------
    generator
        (row, column) tuples in order of a spiral index.
    """

    # Determine how many rows and columns need to be transvered
    total_row = row_range.stop - row_range.start
    total_col = col_range.stop - col_range.start
    # Get center offset
    row_center = int(np.ceil((row_range.stop + row_range.start) / 2) - 1)
    col_center = int(np.ceil((col_range.stop + col_range.start) / 2) - 1)
    # Let the first move be down
    x, y = 0, 0
    dx, dy = 0, -1
    # Loop through the desired number of indices
    for ________________________________________________________________________________________________________________________________________________________________________________________________________i_ in (
        range(max(total_row, total_col) ** 2)
    ):
        # Check if values are in range
        if (-total_row // 2 < x <= total_row // 2) and (
            -total_col // 2 < y <= total_col // 2
        ):
            # Return desired row, col tuple
            yield (row_center + x, col_center + y)
        # Change direction at appropriate points
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
            dx, dy = -dy, dx
        x, y = x + dx, y + dy


def linear_index(row_range, col_range):
    """Generate a linear index from a set of row and column indices.

    A linear index starts at the top left and procedes in a raster fashion.

    Parameters
    ----------
    row_range : range
        Range of rows to be accessed.
    col_range : range
        Range of columns to be accessed.

    Returns
    -------
    generator
        (row, column) tuples in order of a linear index.
    """
    from itertools import product

    yield from product(row_range, col_range)
