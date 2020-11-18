"""OctreeIntersection class.
"""
from typing import List, NamedTuple, Tuple

import numpy as np

from .octree_chunk import OctreeChunk
from .octree_level import OctreeLevel


class OctreeView(NamedTuple):
    """A view into the octree.

    Attributes
    ----------
    corner : np.ndarray
        The two (row, col) corners in data coordinates, base image pixels.
    canvas : np.ndarray
        The shape of the canvas, the window we are drawing into.
    freeze_level : bool
        If True the octree level will not be automatically chosen.
    track_view : bool
        If True which chunks are being rendered should update as the view is moved.
    """

    corners: np.ndarray
    canvas: np.ndarray
    freeze_level: bool
    track_view: bool

    @property
    def data_width(self) -> int:
        """The width between the corners, in data coordinates.

        Return
        ------
            The width in data coordinates.
        """
        return self.corners[1][1] - self.corners[0][1]

    @property
    def auto_level(self) -> bool:
        """True if the octree level should be selected automatically.

        Return
        ------
        bool
            True if the octree level should be selected automatically.
        """
        return not self.freeze_level and self.track_view


class OctreeIntersection:
    """A view's intersection with the octree.

    Parameters
    ----------
    level : OctreeLevel
        The octree level that we intersected with.
    corners_2d : np.ndarray
        The lower left and upper right corners of the view in data coordinates.
    """

    def __init__(self, level: OctreeLevel, view: OctreeView):
        self.level = level

        info = self.level.info

        # TODO_OCTREE: don't split rows/cols so all these pairs of variables
        # are just one variable each? Use numpy more.
        rows, cols = view.corners[:, 0], view.corners[:, 1]

        base = info.slice_config.base_shape

        self.normalized_range = np.array(
            [np.clip(rows / base[0], 0, 1), np.clip(cols / base[1], 0, 1)]
        )

        scaled_rows = rows / info.scale
        scaled_cols = cols / info.scale

        self._row_range = self.row_range(scaled_rows)
        self._col_range = self.column_range(scaled_cols)

    def tile_range(self, span, num_tiles):
        """Return tiles indices needed to draw the span."""

        def _clamp(val, min_val, max_val):
            return max(min(val, max_val), min_val)

        tile_size = self.level.info.slice_config.tile_size

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
        tile_rows = self.level.info.shape_in_tiles[0]
        return self.tile_range(span, tile_rows)

    def column_range(self, span: Tuple[float, float]) -> range:
        """Return column indices which span image coordinates [x0..x1]."""
        tile_cols = self.level.info.shape_in_tiles[1]
        return self.tile_range(span, tile_cols)

    def is_visible(self, row: int, col: int) -> bool:
        """Return True if the tile [row, col] is in the intersection.

        row : int
            The row of the tile.
        col : int
            The col of the tile.
        """

        def _inside(value, value_range):
            return value_range.start <= value < value_range.stop

        return _inside(row, self._row_range) and _inside(col, self._col_range)

    def get_chunks(self, create_chunks=False) -> List[OctreeChunk]:
        """Return chunks inside this intersection.

        Parameters
        ----------
        create_chunks : bool
            If True, create an OctreeChunk at any location that does
            not already have a chunk.
        """
        chunks = []

        # Get every chunk that is within the rectangular region. These are
        # all the chunks we might possibly draw, because they are within
        # the current view.
        #
        # If we've accessed the chunk recently the existing OctreeChunk
        # will be returned, otherwise a new OctreeChunk is created
        # and returned.
        #
        # OctreeChunks can be loaded or unloaded. Unloaded chunks are not
        # drawn until their data as been loaded in. But here we return
        # every chunk within the rectangle.
        for row in self._row_range:
            for col in self._col_range:
                chunk = self.level.get_chunk(
                    row, col, create_chunks=create_chunks
                )
                if chunk is not None:
                    chunks.append(chunk)

        return chunks
