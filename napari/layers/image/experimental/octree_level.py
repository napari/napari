"""OctreeLevel class
"""
from .octree_util import OctreeInfo, OctreeLevelInfo, TileArray


class OctreeLevel:
    """One level of the octree.

    A level contains a 2D array of tiles.

    Soon might also contain a 3D array of sub-volumes.

    Parameters
    ----------
    octree_info : OctreeInfo
        Info that pertains to the entire octree.
    level_index : int
        Index of this specific level (0 is full resolution).
    tile : TileArray
        The tiles for this level.
    """

    def __init__(
        self, octree_info: OctreeInfo, level_index: int, tiles: TileArray
    ):
        shape_in_tiles = [len(tiles), len(tiles[0])]
        self.info = OctreeLevelInfo(octree_info, level_index, shape_in_tiles)
        self.tiles = tiles
