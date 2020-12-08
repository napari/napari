"""Octree class.
"""
import math
from typing import List, Optional

from ....utils.perf import block_timer
from .octree_chunk import OctreeChunk
from .octree_level import OctreeLevel, print_levels
from .octree_tile_builder import create_downsampled_levels
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
    slice_id : int:
        The id of the slice this octree is in.
    data
        The underlying multi-scale data.
    slice_config : SliceConfig
        The base shape and other information.
    """

    def __init__(self, slice_id: int, data, slice_config: SliceConfig):
        self.slice_id = slice_id
        self.data = data
        self.slice_config = slice_config

        _check_downscale_ratio(self.data)  # We expect a ratio of 2.

        self.levels = [
            OctreeLevel(slice_id, data[i], slice_config, i)
            for i in range(len(data))
        ]

        if not self.levels:
            # Probably we will allow empty trees, but for now raise:
            raise ValueError(
                f"Data of shape {data.shape} resulted " "no octree levels?"
            )

        print_levels("Octree input data has", self.levels)

        # If root level contains more than one tile, add extra levels
        # until the root does consist of a single tile. We have to do this
        # because we cannot draw tiles larger than the standard size right now.
        if self.levels[-1].info.num_tiles > 1:
            self.levels.extend(self._get_extra_levels())

        # Now the root should definitely contain only a single tile.
        assert self.levels[-1].info.num_tiles == 1

        # This is now the total number of levels, including the extra ones.
        self.num_levels = len(self.data)

    def get_parent(self, octree_chunk: OctreeChunk) -> Optional[OctreeChunk]:
        """Return the parent of this octree_chunk or None if this is the root.

        Parameters
        ----------
        octree_chunk : OctreeChunk
            Return the parent of this chunk.

        Return
        ------
        Optional[OctreeChunk]
            The parent of the trunk if there was one.
        """
        location = octree_chunk.location

        if location.level_index == self.num_levels - 1:
            return None  # This is the root so no parent.

        row, col = location.row / 2, location.col / 2
        parent_level = self.levels[location.level_index + 1]

        # For now this is None unless there already is an OctreeChunk here.
        return parent_level.get_chunk(row, col)

    def _get_extra_levels(self) -> List[OctreeLevel]:
        """Compute the extra levels and return them.

        Return
        ------
        List[OctreeLevel]
            The extra levels.
        """
        num_levels = len(self.levels)

        with block_timer("_create_extra_levels") as timer:
            extra_levels = self._create_extra_levels(self.slice_id)

        label = f"In {timer.duration_ms:.3f}ms created"
        print_levels(label, extra_levels, num_levels)

        print(f"Tree now has {len(self.levels)} total levels.")

        return extra_levels

    def _create_extra_levels(self, slice_id: int) -> List[OctreeLevel]:
        """Add additional levels to the octree.

        Keep adding levels until we each a root level where the image
        data fits inside a single tile.

        Parameters
        -----------
        slice_id : int
            The id of the slice this octree is in.
        tile_size : int
            Keep creating levels until one fits with a tile of this size.

        Return
        ------
        List[OctreeLevels]
            The new downsampled levels we created.

        Notes
        -----
        If we created this octree data, the root/highest level would
        consist of a single tile. However, if we are reading someone
        else's multiscale data, they did not know our tile size. Or they
        might not have imagined someone using tiled rendering at all.
        So their root level might be pretty large. We've seen root
        levels larger than 8000x8000 pixels.

        It would be nice in this case if our root level could use a
        really large tile size as a special case. So small tiles for most
        levels, but then one big tile as the root.

        However, today our TiledImageVisual can't handle that. It can't
        handle a mix of tile sizes, and TextureAtlas2D can't even
        allocate multiple textures!

        So for now, we compute/downsample additional levels ourselves and
        add them to the data. We do this until we reach a level that does
        fit within a single tile.

        For example for that 8000x8000 pixel root level, if we are using
        256x256 tiles we'll keep adding levels until we get to a level
        that fits within 256x256.

        Unfortunately, we can't be sure our downsampling approach will
        match the rest of the data. The levels we add might be more/less
        blurry than the original ones, or have different downsampling
        artifacts. That's probably okay because these are low-resolution
        levels, but still it would be better to show their root level
        verbatim on a single large tiles.

        Also, this is slow due to the downsampling, but also slow due to
        having to load the full root level into memory. If we had a root
        level that was tile sized, then we could load that very quickly.
        That issue does not go away even if we have a special large-size
        root tile. For best performance multiscale data should be built
        down to a very small root tile, the extra storage is negligible.
        """

        # Create additional data levels so that the root level
        # consists of only a single tile, using our standard/only
        # tile size.
        tile_size = self.slice_config.tile_size
        new_levels = create_downsampled_levels(self.data[-1], tile_size)

        # Add the data.
        self.data.extend(new_levels)

        # Return an OctreeLevel for each new data level.
        num_current = len(self.levels)
        return [
            OctreeLevel(
                slice_id, new_data, self.slice_config, num_current + index,
            )
            for index, new_data in enumerate(new_levels)
        ]

    def print_info(self):
        """Print information about our tiles."""
        for level in self.levels:
            level.print_info()


def _check_downscale_ratio(data) -> None:
    """Raise exception if downscale ratio is not 2.

    For now we only support downscale ratios of 2. We could support other
    ratios, but the assumption that each octree level is half the size of
    the previous one is baked in pretty deeply right now.

    Raises
    ------
    ValueError
        If downscale ratio is not 2.
    """
    if not isinstance(data, list) or len(data) < 2:
        return  # There aren't even two levels.

    ratio = math.sqrt(data[0].size / data[1].size)

    # Really should be exact, but it will most likely be off by a ton
    # if its off, so allow a small fudge factor.
    if not math.isclose(ratio, 2, rel_tol=0.01):
        raise ValueError(
            f"Multiscale data has downsampling ratio of {ratio}, expected 2."
        )
