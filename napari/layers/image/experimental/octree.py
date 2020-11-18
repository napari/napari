"""Octree class.
"""
from typing import List

from ...._vendor.experimental.humanize.src.humanize import intword
from ....utils.perf import block_timer
from .octree_level import OctreeLevel
from .octree_tile_builder import create_downsampled_levels
from .octree_util import SliceConfig


def _dim_str(dim: tuple) -> None:
    return f"{dim[0]} x {dim[1]} = {intword(dim[0] * dim[1])}"


def _print_levels(
    label: str, levels: List[OctreeLevel], start: int = 0
) -> None:
    print(f"{label} {len(levels)} levels:")
    for i, level in enumerate(levels):
        image_str = _dim_str(level.info.image_shape)
        tiles_str = _dim_str(level.info.shape_in_tiles)
        level = start + i
        print(f"    Level {level}: {image_str} pixels -> {tiles_str} tiles")


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

        if not self.levels:
            # Probably we will allow empty trees, but for now raise:
            raise ValueError(
                f"Data of shape {data.shape} resulted " "no octree levels?"
            )

        _print_levels("Octree input data has", self.levels)
        original_levels = len(self.levels)

        # If root level contains more than one tile, add more levels
        # until the root does consist of a single tile.
        if self.levels[-1].info.num_tiles > 1:
            with block_timer("_create_additional_levels") as timer:
                more_levels = self._create_additional_levels(slice_id)

            _print_levels(
                f"In {timer.duration_ms:.3f}ms created",
                more_levels,
                start=original_levels,
            )
            self.levels.extend(more_levels)

            print(f"Tree now has {len(self.levels)} total levels.")

        # Now the root should definitely contain only a single tile.
        assert self.levels[-1].info.num_tiles == 1

        # This now the total number of levels.
        self.num_levels = len(data)

    def _create_additional_levels(self, slice_id: int) -> List[OctreeLevel]:
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
