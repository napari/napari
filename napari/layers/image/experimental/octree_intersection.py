"""OctreeView and OctreeIntersection classes.
"""
from typing import List, NamedTuple, Tuple

import numpy as np

from napari.layers.image.experimental.octree_chunk import OctreeChunk
from napari.layers.image.experimental.octree_level import OctreeLevel
from napari.layers.image.experimental.octree_util import (
    OctreeDisplayOptions,
    spiral_index,
)

MAX_NUM_CHUNKS = 81


class OctreeView(NamedTuple):
    """A view into the octree.

    An OctreeView corresponds to a camera which is viewing the image data,
    plus options as to how we want to render the data.

    Attributes
    ----------
    corner : np.ndarray
        The two (row, col) corners in data coordinates, base image pixels.
    canvas : np.ndarray
        The shape of the canvas, the window we are drawing into.
    display : OctreeDisplayOptions
        How to display the view.
    """

    corners: np.ndarray
    canvas: np.ndarray
    display: OctreeDisplayOptions

    @property
    def data_width(self) -> int:
        """The width between the corners, in data coordinates.

        Returns
        -------
        int
            The width in data coordinates.
        """
        return self.corners[1][1] - self.corners[0][1]

    @property
    def auto_level(self) -> bool:
        """True if the octree level should be selected automatically.

        Returns
        -------
        bool
            True if the octree level should be selected automatically.
        """
        return not self.display.freeze_level and self.display.track_view

    def expand(self, expansion_factor: float) -> 'OctreeView':
        """Return expanded view.

        We expand the view so that load some tiles around the edge, so if
        you pan they are more likely to be already loaded.

        Parameters
        ----------
        expansion_factor : float
            Expand the view by this much. Contract if less than 1.
        """
        assert expansion_factor > 0

        extents = self.corners[1] - self.corners[0]
        padding = ((extents * expansion_factor) - extents) / 2
        new_corners = np.array(
            (self.corners[0] - padding, self.corners[1] + padding)
        )
        return OctreeView(new_corners, self.canvas, self.display)


class OctreeIntersection:
    """A view's intersection with the octree.

    Parameters
    ----------
    level : OctreeLevel
        The octree level that we intersected with.
    view : OctreeView
        The view we are intersecting with the octree.
    """

    def __init__(self, level: OctreeLevel, view: OctreeView) -> None:
        self.level = level
        self._corners = view.corners

        level_info = self.level.info

        # TODO_OCTREE: don't split rows/cols so all these pairs of variables
        # are just one variable each? Use numpy more.
        rows, cols = view.corners[:, 0], view.corners[:, 1]

        base = level_info.meta.base_shape

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

        tile_size = self.level.info.meta.tile_size

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

        Returns
        -------
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

        Returns
        -------
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
        # the chunks we want to draw to depict this region of the data.
        #
        # If we've accessed the chunk recently the existing OctreeChunk
        # will be returned, otherwise a new OctreeChunk is created
        # and returned.
        #
        # OctreeChunks can be loaded or unloaded. Unloaded chunks are not
        # drawn until their data as been loaded in. But here we return
        # every chunk within the view.

        # We use spiral indexing to get chunks from the center first
        for i, (row, col) in enumerate(
            spiral_index(self._row_range, self._col_range)
        ):
            chunk = self.level.get_chunk(row, col, create=create)
            if chunk is not None:
                chunks.append(chunk)
            # We place a limit on the maximum number of chunks that
            # we'll ever take from a level to deal with the single
            # level tiled rendering case.
            if i > MAX_NUM_CHUNKS:
                break
        return chunks

    @property
    def tile_state(self) -> dict:
        """Return tile state, for the monitor.

        Returns
        -------
        dict
            The tile state.
        """
        x, y = np.mgrid[self._row_range, self._col_range]
        seen = np.vstack((x.ravel(), y.ravel())).T

        return {
            # A list of (row, col) pairs of visible tiles.
            "seen": seen,
            # The two corners of the view in data coordinates ((x0, y0), (x1, y1)).
            "corners": self._corners,
        }

    @property
    def tile_config(self) -> dict:
        """Return tile config, for the monitor.

        Returns
        -------
        dict
            The file config.
        """
        # TODO_OCTREE: Need to cleanup and re-name and organize
        # OctreeLevelInfo and OctreeMetadata attrbiutes. Messy.
        level = self.level
        image_shape = level.info.image_shape
        shape_in_tiles = level.info.shape_in_tiles

        meta = level.info.meta
        base_shape = meta.base_shape
        tile_size = meta.tile_size

        return {
            "base_shape": base_shape,
            "image_shape": image_shape,
            "shape_in_tiles": shape_in_tiles,
            "tile_size": tile_size,
            "level_index": level.info.level_index,
        }
