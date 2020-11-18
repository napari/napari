"""Octree class.
"""
from .octree_level import OctreeLevel
from .octree_util import SliceConfig


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

    def __init__(self, slice_id: int, data, slice_config: SliceConfig):
        self.data = data
        self.slice_config = slice_config

        self.levels = [
            OctreeLevel(slice_id, data[i], slice_config, i)
            for i in range(len(data))
        ]
        self.num_levels = len(data)

    def print_info(self):
        """Print information about our tiles."""
        for level in self.levels:
            level.print_info()
