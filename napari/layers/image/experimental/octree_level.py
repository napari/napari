"""OctreeLevel class
"""
from .octree_util import OctreeInfo, OctreeLevelInfo, TileArray


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
