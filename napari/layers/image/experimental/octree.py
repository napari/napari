"""Octree class.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Optional

from napari.layers.image.experimental.octree_level import (
    OctreeLevel,
    log_levels,
)
from napari.layers.image.experimental.octree_tile_builder import (
    create_downsampled_levels,
)
from napari.layers.image.experimental.octree_util import OctreeMetadata
from napari.utils.perf import block_timer
from napari.utils.translations import trans

LOGGER = logging.getLogger("napari.octree")

if TYPE_CHECKING:
    from napari.components.experimental.chunk._request import OctreeLocation
    from napari.layers.image.experimental.octree_chunk import OctreeChunk


class Octree:
    """A sparse region octree that holds hold 2D or 3D images.

    The Octree is sparse in that it contains the ArrayLike data for the
    image, but it does not contain chunks or nodes or anything for that
    data. There is no actual tree for the data, only the multiscale data
    itself.

    Instead Octree just provides methods so that other classes like
    OctreeLevel, OctreeSlice and OctreeLoader can create and store OctreeChunks
    for just the portion of the tree we are rendering. This class knows
    how to go from chunks to their parents or children, but those parents
    or children are only created as needed.

    Notes
    -----
    Future work related to geometry: Eventually we want our octree to hold
    geometry, not just images. Geometry such as points and meshes. For
    geometry a sparse octree might make more sense than this full/complete
    region octree.

    With geometry there might be lots of empty space in between small dense
    pockets of geometry. Some parts of tree might need to be very deep, but
    it would be a waste for the tree to be that deep everywhere.

    Parameters
    ----------
    slice_id : int:
        The id of the slice this octree is in.
    data
        The underlying multi-scale data.
    meta : OctreeMetadata
        The base shape and other information.
    """

    def __init__(self, slice_id: int, data, meta: OctreeMetadata) -> None:
        self.slice_id = slice_id
        self.data = data
        self.meta = meta

        _check_downscale_ratio(self.data)  # We expect a ratio of 2.

        self.levels = [
            OctreeLevel(slice_id, data[i], meta, i) for i in range(len(data))
        ]

        if not self.levels:
            # Probably we will allow empty trees, but for now raise:
            raise ValueError(
                trans._(
                    "Data of shape {shape} resulted no octree levels?",
                    deferred=True,
                    shape=data.shape,
                )
            )

        LOGGER.info("Multiscale data has %d levels.", len(self.levels))

        # If there is more than one level and the root level contains more
        # than one tile, add extra levels until the root does consist of a
        # single tile. We have to do this because we cannot draw tiles larger
        # than the standard size right now.
        # If there is only one level than we'll only ever be able to show tiles
        # from that level.
        if len(self.data) > 1 and self.levels[-1].info.num_tiles > 1:
            self.levels.extend(self._get_extra_levels())

        LOGGER.info("Octree now has %d total levels:", len(self.levels))
        log_levels(self.levels)

        # Now the root should definitely contain only a single tile if there is
        # more than one level
        if len(self.data) > 1:
            assert self.levels[-1].info.num_tiles == 1

        # This is now the total number of levels, including the extra ones.
        self.num_levels = len(self.data)

    def get_level(self, level_index: int) -> OctreeLevel:
        """Get the given OctreeLevel.

        Parameters
        ----------
        level_index : int
            Get the OctreeLevel with this index.

        Returns
        -------
        OctreeLevel
            The requested level.
        """
        try:
            return self.levels[level_index]
        except IndexError as exc:
            raise IndexError(
                trans._(
                    "Level {level_index} is not in range(0, {top})",
                    deferred=True,
                    level_index=level_index,
                    top=len(self.levels),
                )
            ) from exc

    def get_chunk_at_location(
        self, location: OctreeLocation, create: bool = False
    ) -> None:
        """Get chunk get this location, create if needed if create=True.

        Parameters
        ----------
        location : OctreeLocation
            Get chunk at this location.
        create : bool
            If True create the chunk if it doesn't exist.
        """
        return self.get_chunk(
            location.level_index, location.row, location.col, create=create
        )

    def get_chunk(
        self, level_index: int, row: int, col: int, create=False
    ) -> Optional[OctreeChunk]:
        """Get chunk at this location, create if needed if create=True

        Parameters
        ----------
        level_index : int
            Get chunk from this level.
        row : int
            Get chunk at this row.
        col : int
            Get chunk at this col.
        """
        level: OctreeLevel = self.get_level(level_index)
        return level.get_chunk(row, col, create=create)

    def get_parent(
        self,
        octree_chunk: OctreeChunk,
        create: bool = False,
    ) -> Optional[OctreeChunk]:
        """Return the parent of this octree_chunk.

        If the chunk at the root level, then this always return None.
        Otherwise it returns the parent if it exists, or if create=True
        it creates the parent and returns it.

        Parameters
        ----------
        octree_chunk : OctreeChunk
            Return the parent of this chunk.

        Returns
        -------
        Optional[OctreeChunk]
            The parent of the chunk if there was one or we created it.
        """
        ancestors = self.get_ancestors(octree_chunk, 1, create=create)
        # If no parent exists yet then returns None
        if len(ancestors) == 0:
            return None
        else:
            return ancestors[0]

    def get_ancestors(
        self,
        octree_chunk: OctreeChunk,
        num_levels=None,
        create=False,
        in_memory: bool = False,
    ) -> List[OctreeChunk]:
        """Return the num_levels nearest ancestors.

        If create=False only returns the ancestors if they exist. If
        create=True it will create the ancestors as needed.

        Parameters
        ----------
        octree_chunk : OctreeChunk
            Return the nearest ancestors of this chunk.
        num_levels : int, optional
            Number of levels to look. If not provided then all are looked back till
            the root level.
        create : bool
            Whether to create the chunk of not is it doesn't exist.
        in_memory : bool
            Whether to return only in memory chunks or not.

        Returns
        -------
        List[OctreeChunk]
            Up to num_level nearest ancestors of the given chunk. Sorted so the
            most-distant ancestor comes first.
        """
        ancestors = []
        location = octree_chunk.location

        # Starting point, we look up from here.
        level_index = location.level_index
        row, col = location.row, location.col

        if num_levels is None:
            stop_level = self.num_levels - 1
        else:
            stop_level = min(self.num_levels - 1, level_index + num_levels)

        # Search up one level at a time.
        while level_index < stop_level:

            # Get the next level up. Coords are halved each level.
            level_index += 1
            row, col = int(row / 2), int(col / 2)

            # Get chunk at this location.
            ancestor = self.get_chunk(level_index, row, col, create=create)
            if create:
                assert ancestor  # Since create=True
            ancestors.append(ancestor)

        # Keep non-None children, and if requested in-memory ones.
        def keep_chunk(octree_chunk) -> bool:
            return octree_chunk is not None and (
                not in_memory or octree_chunk.in_memory
            )

        # Reverse to provide the most distant ancestor first.
        return list(filter(keep_chunk, reversed(ancestors)))

    def get_children(
        self,
        octree_chunk: OctreeChunk,
        create: bool = False,
        in_memory: bool = True,
    ) -> List[OctreeChunk]:
        """Return the children of this octree_chunk.

        If create is False then we only return children that exist, so we will
        return between 0 and 4 children. If create is True then we will create
        any children that don't exist. If octree_chunk is in level 0 then
        we will always return 0 children.

        Parameters
        ----------
        octree_chunk : OctreeChunk
            Return the children of this chunk.

        Returns
        -------
        List[OctreeChunk]
            The children of the given chunk.
        """
        location = octree_chunk.location

        if location.level_index == 0:
            return []  # This is the base level so no children.

        child_level_index: int = location.level_index - 1
        child_level: OctreeLevel = self.levels[child_level_index]

        row, col = location.row * 2, location.col * 2

        children = [
            child_level.get_chunk(row, col, create=create),
            child_level.get_chunk(row, col + 1, create=create),
            child_level.get_chunk(row + 1, col, create=create),
            child_level.get_chunk(row + 1, col + 1, create=create),
        ]

        # Keep non-None children, and if requested in-memory ones.
        def keep_chunk(octree_chunk) -> bool:
            return octree_chunk is not None and (
                not in_memory or octree_chunk.in_memory
            )

        return list(filter(keep_chunk, children))

    def _get_extra_levels(self) -> List[OctreeLevel]:
        """Compute the extra levels and return them.

        Returns
        -------
        List[OctreeLevel]
            The extra levels.
        """
        with block_timer("_create_extra_levels") as timer:
            extra_levels = self._create_extra_levels(self.slice_id)

        LOGGER.info(
            "Created %d additional levels in %.3fms",
            len(extra_levels),
            timer.duration_ms,
        )

        return extra_levels

    def _create_extra_levels(self, slice_id: int) -> List[OctreeLevel]:
        """Add additional levels to the octree.

        Keep adding levels until we each a root level where the image
        data fits inside a single tile.

        Parameters
        ----------
        slice_id : int
            The id of the slice this octree is in.

        Returns
        -------
        List[OctreeLevels]
            The new downsampled levels we created.

        Notes
        -----
        Whoever created this multiscale data probably did not know our tile
        size. So their root level might be pretty large. We've seen root
        levels larger than 8000x8000 pixels.

        Since our visual can't have variable size tiles or large tiles yet,
        we compute/downsample additional levels ourselves and add them to
        the data. We do this until we reach a level that fits within a
        single tile.

        For example, for that 8000x8000 pixel root level, if we are using
        (256, 256) tiles we'll keep adding levels until we get to a level
        that fits within (256, 256).

        Unfortunately, we can't be sure our downsampling approach will
        visually match the rest of the data. That's probably okay because
        these are the lowest-resolution levels. But this another reason
        it'd be better if our visuals could draw large tiles when needed.

        Also, downsampling can be very slow.
        """

        # Create additional data levels so that the root level
        # consists of only a single tile, using our standard/only
        # tile size.
        tile_size = self.meta.tile_size
        new_levels = create_downsampled_levels(
            self.data[-1], len(self.data), tile_size
        )

        # Add the data.
        self.data.extend(new_levels)

        # Return an OctreeLevel for each new data level.
        num_current = len(self.levels)
        return [
            OctreeLevel(
                slice_id,
                new_data,
                self.meta,
                num_current + index,
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

    # _dump_levels(data)

    ratio = math.sqrt(data[0].size / data[1].size)

    # Really should be exact, but it will most likely be off by a ton
    # if its off, so allow a small fudge factor.
    if not math.isclose(ratio, 2, rel_tol=0.01):
        raise ValueError(
            trans._(
                "Multiscale data has downsampling ratio of {ratio}, expected 2.",
                deferred=True,
                ratio=ratio,
            )
        )


def _dump_levels(data) -> None:
    """Print the levels and the size of the levels."""
    last_size = None
    for level in data:
        if last_size is not None:
            downscale = math.sqrt(last_size / level.size)
            print(
                f"size={level.size} shape={level.shape} downscale={downscale}"
            )
        else:
            print(f"size={level.size} shape={level.shape} base level")
        last_size = level.size
