"""OctreeLevel class
"""
from typing import List

from .octree_util import ChunkData, OctreeInfo, OctreeIntersection, TileArray


class OctreeLevel:
    """One level of the octree.

    A level contains a 2D or 3D array of tiles.
    """

    def __init__(self, info: OctreeInfo, level_index: int, tiles: TileArray):
        self.info = info
        self.level_index = level_index
        self.tiles = tiles

        self.scale = 2 ** self.level_index
        self.num_rows = len(self.tiles)
        self.num_cols = len(self.tiles[0])

    def print_info(self):
        """Print information about this level."""
        nrows = len(self.tiles)
        ncols = len(self.tiles[0])
        print(f"level={self.level_index} dim={nrows}x{ncols}")

    def get_intersection(self, data_corners) -> OctreeIntersection:

        # TODO_OCTREE: we should scale the corners somewhere else?
        data_corners /= self.scale

        # TODO_OCTREE: fix this for any dims
        data_rows = [data_corners[0][1], data_corners[1][1]]
        data_cols = [data_corners[0][2], data_corners[1][2]]

        row_range = self.row_range(data_rows)
        col_range = self.column_range(data_cols)

        return OctreeIntersection(
            [self.num_rows, self.num_cols], [row_range, col_range]
        )

    def get_chunks(self, data_corners) -> List[ChunkData]:
        """Return chunks that are within this rectangular region of the data.

        Parameters
        ----------
        data_corners
            Return chunks within this rectangular region.
        """
        chunks = []

        intersection = self.get_intersection(data_corners)

        scale = self.scale
        scale_vec = [scale, scale]

        tile_size = self.info.tile_size

        ranges = intersection.ranges

        # Iterate over every tile in the rectangular region.
        data = None
        y = ranges[0].start * tile_size
        for row in ranges[0]:
            x = ranges[1].start * tile_size
            for col in ranges[1]:

                data = self.tiles[row][col]
                pos = [x, y]

                if 0 not in data.shape:
                    chunks.append(ChunkData(data, pos, scale_vec))

                x += data.shape[1] * scale
            y += data.shape[0] * scale

        return chunks

    def tile_range(self, span, num_tiles):
        """Return tiles indices needed to draw the span."""

        def _clamp(val, min_val, max_val):
            return max(min(val, max_val), min_val)

        tile_size = self.info.tile_size
        print(tile_size)

        tiles = [span[0] / tile_size, span[1] / tile_size]
        print(f"tiles = {tiles}")
        new_min = _clamp(tiles[0], 0, num_tiles - 1)
        new_max = _clamp(tiles[1], 0, num_tiles - 1)
        clamped = [new_min, new_max + 1]
        print(f"clamped = {clamped}")

        span_int = [int(x) for x in clamped]
        return range(*span_int)

    def row_range(self, span):
        """Return row indices which span image coordinates [y0..y1]."""
        return self.tile_range(span, self.num_rows)

    def column_range(self, span):
        """Return column indices which span image coordinates [x0..x1]."""
        return self.tile_range(span, self.num_cols)
