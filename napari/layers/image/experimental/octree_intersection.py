"""OctreeIntersection class.
"""
from typing import Tuple

import numpy as np

from .octree_util import OctreeLevelInfo

# TODO_OCTREE: These types might be a horrible idea but trying it for now.
Float2 = np.ndarray  # [x, y] dtype=float64 (default type)


class OctreeIntersection:
    # TODO_OCTREE: this class needs a lot of work

    def __init__(self, info: OctreeLevelInfo, data_corners):
        self.info = info
        self.data_corners = data_corners

        # TODO_OCTREE: don't split rows/cols?
        self.rows: Float2 = data_corners[:, 0]
        self.cols: Float2 = data_corners[:, 1]

        base = self.info.octree_info.base_shape

        self.normalized_rows = np.clip(self.rows / base[0], 0, 1)
        self.normalized_cols = np.clip(self.cols / base[1], 0, 1)

        self.rows /= self.info.scale
        self.cols /= self.info.scale

        self.row_range = self.row_range(self.rows)
        self.col_range = self.column_range(self.cols)

    def tile_range(self, span, num_tiles):
        """Return tiles indices needed to draw the span."""

        def _clamp(val, min_val, max_val):
            return max(min(val, max_val), min_val)

        tile_size = self.info.octree_info.tile_size

        span_tiles = [span[0] / tile_size, span[1] / tile_size]
        clamped = [
            _clamp(span_tiles[0], 0, num_tiles - 1),
            _clamp(span_tiles[1], 0, num_tiles - 1) + 1,
        ]

        # int() truncates which is what we want
        span_int = [int(x) for x in clamped]
        return range(*span_int)

    def row_range(self, span: Tuple[float, float]) -> range:
        """Return row indices which span image coordinates [y0..y1]."""
        return self.tile_range(span, self.info.tile_shape[0])

    def column_range(self, span: Tuple[float, float]) -> range:
        """Return column indices which span image coordinates [x0..x1]."""
        return self.tile_range(span, self.info.tile_shape[1])

    def is_visible(self, row: int, col: int) -> bool:
        """Return True if the tile [row, col] is in the intersection.

        row : int
            The row of the tile.
        col : int
            The col of the tile.
        """

        def _inside(value, value_range):
            return value >= value_range.start and value < value_range.stop

        return _inside(row, self.row_range) and _inside(col, self.col_range)
