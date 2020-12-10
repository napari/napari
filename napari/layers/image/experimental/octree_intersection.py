"""OctreeIntersection class.
"""
from typing import List, NamedTuple, Tuple

import numpy as np

from .octree_chunk import OctreeChunk
from .octree_level import OctreeLevel
from .octree_util import OctreeDisplayOptions


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
    display: OctreeDisplayOptions

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
        return not self.display.freeze_level and self.display.track_view


class OctreeIntersection:
    """A view's intersection with the octree.

    Parameters
    ----------
    level : OctreeLevel
        The octree level that we intersected with.
    view : OctreeView
        The view we are intersecting with the octree.
    """

    def __init__(self, level: OctreeLevel, view: OctreeView):
        self.level = level
        self._corners = view.corners

        level_info = self.level.info

        # TODO_OCTREE: don't split rows/cols so all these pairs of variables
        # are just one variable each? Use numpy more.
        rows, cols = view.corners[:, 0], view.corners[:, 1]

        base = level_info.slice_config.base_shape

        self.normalized_range = np.array(
            [np.clip(rows / base[0], 0, 1), np.clip(cols / base[1], 0, 1)]
        )

        scaled_rows = rows / level_info.scale
        scaled_cols = cols / level_info.scale

        self._row_range = self.row_range(scaled_rows)
        self._col_range = self.column_range(scaled_cols)

    def tile_range(
        self, span: Tuple[float, float], num_tiles_total: int
    ) -> range:
        """Return tiles indices needed to draw the span.

        Parameters
        ----------
        span : Tuple[float, float]
            The span in image coordinates.
        num_tiles_total : int
            The total number of tiles in this direction.
        """

        def _clamp(val, min_val, max_val):
            return max(min(val, max_val), min_val)

        tile_size = self.level.info.slice_config.tile_size

        span_tiles = [span[0] / tile_size, span[1] / tile_size]
        clamped = [
            _clamp(span_tiles[0], 0, num_tiles_total - 1),
            _clamp(span_tiles[1], 0, num_tiles_total - 1) + 1,
        ]

        # TODO_OCTREE: BUG, range is not empty when it should be?

        # int() truncates which is what we want
        span_int = [int(x) for x in clamped]
        return range(*span_int)

    def row_range(self, span: Tuple[float, float]) -> range:
        """Return row range of tiles for this span.
        Parameters
        ----------
        span : Tuple[float, float]
            The span in image coordinates, [y0..y1]

        Return
        ------
        range
            The range of tiles across the columns.
        """
        tile_rows = self.level.info.shape_in_tiles[0]
        return self.tile_range(span, tile_rows)

    def column_range(self, span: Tuple[float, float]) -> range:
        """Return column range of tiles for this span.

        Parameters
        ----------
        span : Tuple[float, float]
            The span in image coordinates, [x0..x1]

        Return
        ------
        range
            The range of tiles across the columns.
        """
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

    def get_chunks(self, create=False) -> List[OctreeChunk]:
        """Return all of the chunks in this intersection.

        Parameters
        ----------
        create : bool
            If True, create an OctreeChunk at any location that does
            not already have one.
        """
        chunks = []  # The chunks in the intersection.

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
        # every chunk within the view.
        for row in self._row_range:
            for col in self._col_range:
                chunk = self.level.get_chunk(row, col, create=create)
                if chunk is not None:
                    chunks.append(chunk)

        return chunks

    @property
    def tile_state(self) -> dict:
        """Return tile state, for the monitor.

        Return
        ------
        dict
            The tile state.
        """
        x, y = np.mgrid[self._row_range, self._col_range]
        seen = np.vstack((x.ravel(), y.ravel())).T

        return {
            "tile_state": {
                # A list of (row, col) pairs of visible tiles.
                "seen": seen,
                # The two corners of the view in data coordinates ((x0, y0), (x1, y1)).
                "corners": self._corners,
            }
        }

    @property
    def tile_config(self) -> dict:
        """Return tile config, for the monitor.

        Return
        ------
        dict
            The file config.
        """
        # TODO_OCTREE: Need to cleanup and re-name and organize
        # OctreeLevelInfo and SliceConfig attrbiutes. Messy.
        level = self.level
        image_shape = level.info.image_shape
        shape_in_tiles = level.info.shape_in_tiles

        slice_config = level.info.slice_config
        base_shape = slice_config.base_shape
        tile_size = slice_config.tile_size

        return {
            "tile_config": {
                "base_shape": base_shape,
                "image_shape": image_shape,
                "shape_in_tiles": shape_in_tiles,
                "tile_size": tile_size,
                "level_index": level.info.level_index,
            }
        }
