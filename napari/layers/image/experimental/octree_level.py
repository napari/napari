"""OctreeLevel class
"""
from typing import List

from .octree_intersection import OctreeIntersection
from .octree_util import ChunkData, OctreeInfo, OctreeLevelInfo, TileArray


class OctreeLevel:
    """One level of the octree.

    A level contains a 2D array of tiles.

    Soon might also contain a 3D array of sub-volumes.
    """

    def __init__(
        self, octree_info: OctreeInfo, level_index: int, tiles: TileArray
    ):
        self.info = OctreeLevelInfo(
            octree_info, level_index, [len(tiles), len(tiles[0])]
        )

        self.tiles = tiles

    def print_info(self):
        """Print information about this level."""
        nrows = len(self.tiles)
        ncols = len(self.tiles[0])
        print(f"level={self.level_index} dim={nrows}x{ncols}")

    def get_intersection(self, data_corners) -> OctreeIntersection:

        return OctreeIntersection(self.info, data_corners)

    def get_chunks(self, data_corners) -> List[ChunkData]:
        """Return chunks that are within this rectangular region of the data.

        Parameters
        ----------
        data_corners
            Return chunks within this rectangular region.
        """
        chunks = []

        intersection = self.get_intersection(data_corners)

        scale = self.info.scale
        scale_vec = [scale, scale]

        tile_size = self.info.octree_info.tile_size

        # Iterate over every tile in the rectangular region.
        data = None
        y = intersection.row_range.start * tile_size
        for row in intersection.row_range:
            x = intersection.col_range.start * tile_size
            for col in intersection.col_range:

                data = self.tiles[row][col]
                pos = [x, y]

                # Skip tiles with zero area (why are there any?)
                if 0 not in data.shape:
                    print(f"pos={pos}")
                    level_index = self.info.level_index
                    chunks.append(ChunkData(level_index, data, pos, scale_vec))

                x += data.shape[1] * scale
            y += data.shape[0] * scale

        return chunks
