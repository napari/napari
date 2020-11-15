"""Octree class.
"""
from typing import List

import numpy as np

from .octree_level import OctreeLevel
from .octree_tile_builder import create_multi_scale_levels
from .octree_util import OctreeInfo, TileArray

Levels = List[TileArray]


class Octree:
    """A region octree that holds hold 2D or 3D images.

    Today the octree is full/complete meaning every node has 4 or 8
    children, and every leaf node is at the same level of the tree. This
    makes sense for region/image trees, because the image exists
    everywhere.

    Since we are a complete tree we don't need actual nodes with references
    to the node's children. Instead, every level is just an array, and
    going from parent to child or child to parent is trivial, you just
    need to double or half the indexes.

    Future Work: Geometry
    ---------------------
    Eventually we want our octree to hold geometry, not just images.
    Geometry such as points and meshes. For geometry a sparse octree might
    make more sense than this full/complete region octree.

    With geometry there might be lots of empty space in between small dense
    pockets of geometry. Some parts of tree might need to be very deep, but
    it would be a waste for the tree to be that deep everywhere.

    Parameters
    ----------
    base_shape : Tuple[int, int]
        The shape of the full base image.
    levels : Levels
        All the levels of the tree.
    """

    def __init__(self, info: OctreeInfo, levels: Levels):
        self.info = info
        self.levels = [
            OctreeLevel(info, i, level) for (i, level) in enumerate(levels)
        ]
        self.num_levels = len(self.levels)  # move to self.info?

    def print_info(self):
        """Print information about our tiles."""
        for level in self.levels:
            level.print_info()

    @classmethod
    def from_image(cls, image: np.ndarray, tile_size: int):
        """Create octree from given single image.

        Parameters
        ----------
        image : ndarray
            Create the octree for this single image.
        """
        levels = create_multi_scale_levels(image, tile_size)

        info = OctreeInfo.create(image.shape, tile_size)
        return Octree(info, levels)
