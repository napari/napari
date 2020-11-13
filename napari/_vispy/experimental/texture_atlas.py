"""TextureAtlas2D class.

A texture atlas is a large texture that stores many smaller texture tiles.
"""
from typing import NamedTuple, Optional, Tuple

import numpy as np
from vispy.gloo import Texture2D

# Two triangles which cover a [0..1, 0..1] quad.
_QUAD = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]], dtype=np.float32,
)


class AtlasTile(NamedTuple):
    """Information about one specific tile in the atlas.

    AtlasTile is returned from TextureAtlas2D.add_tile() so the caller has
    the texture coordinates to render each tile in the atlas.
    """

    index: int
    tex_coords: np.ndarray


class TileSpec(NamedTuple):
    """Information about the tiles we are using in the atlas."""

    shape: np.ndarray
    ndim: int
    height: int
    width: int
    depth: int

    @classmethod
    def from_shape(cls, shape: np.ndarray):
        """Create a TileInfo from just the shape."""
        ndim = len(shape)
        assert ndim in [2, 3]  # 2D or 2D with color.
        height, width = shape[:2]
        depth = 1 if ndim == 2 else shape[2]
        return cls(shape, ndim, height, width, depth)

    def is_compatible(self, data: np.ndarray) -> bool:
        """Return True if the given data is compatible with our tiles.

        Parameters
        ----------
        data : np.ndarray
            Return True if this data is compatible with our tiles.
        """
        if self.ndim != data.ndim:
            return False  # Different number of dimensions.

        if self.ndim == 3 and self.depth != data.shape[2]:
            return False  # Different depths.

        if data.shape[0] > self.height or data.shape[1] > self.width:
            return False  # Data is too big for the tile.

        # It's either an exact match, or it's compatible but smaller than
        # the full tile, which is fine.
        return True


class TextureAtlas2D(Texture2D):
    """A two-dimensional texture atlas.

    A single large texture with "slots" for smaller texture tiles.

    Parameters
    ----------
    tile_shape : tuple
        The (height, width) of one tile in texels.
    texture_shape : Tuple[int, int]
        The (height, width) of the full texture in terms of tiles.
    """

    # If True we mark removed tiles red for debugging. When we remove a
    # tile we do not need to waste bandwidth clearing the old texture data.
    # We just let that slot get written to later on. However for debugging
    # we can write into that spot to mark it as empty.
    MARK_DELETED_TILES = False

    def __init__(
        self, tile_shape: tuple, shape_in_tiles: Tuple[int, int], **kwargs,
    ):
        # Each tile's shape in texels, for example (256, 256, 3).
        self.spec = TileSpec.from_shape(tile_shape)

        # The full texture's shape in tiles, for example 4x4.
        self.shape_in_tiles = shape_in_tiles

        depth = 3  # TODO_OCTREE: get from the data

        # The full texture's shape in texels, for example 1024x1024.
        height = self.spec.height * self.shape_in_tiles[0]
        width = self.spec.width * self.shape_in_tiles[1]
        self.texture_shape = np.array([width, height, depth], dtype=np.int32)

        # Total number of texture slots in the atlas.
        self.num_slots_total = shape_in_tiles[0] * shape_in_tiles[1]

        # Every index is free initially.
        self._free_indices = set(range(0, self.num_slots_total))

        # Pre-compute the texture coords for every tile. Otherwise we'd be
        # calculating these over and over as tiles are added. These are for
        # full tiles only. Edge and corner tiles will need custom texture
        # coordinates based on their size.
        #
        # TODO_OCTREE: Put all these into one compact ndarray?
        self._tex_coords = [
            self._calc_tex_coords(i) for i in range(self.num_slots_total)
        ]

        if self.MARK_DELETED_TILES:
            shape = self.spec.shape
            self.deleted_tile_data = np.empty(shape, dtype=np.uint8)
            self.deleted_tile_data[:] = (1, 1, 1)  # handle RGB or RGBA?

        super().__init__(shape=tuple(self.texture_shape), **kwargs)

    @property
    def num_slots_free(self) -> int:
        """Return the number of available texture slots.

        Return
        ------
        int
            The number of available texture slots.
        """
        return len(self._free_indices)

    @property
    def num_slots_used(self) -> int:
        """Return the number of used texture slots.

        Return
        ------
        int
            The number of used texture slots.
        """
        return self.num_slots_total - self.num_slots_free

    def _offset(self, tile_index: int) -> Tuple[int, int]:
        """Return the (X, Y) offset into the full atlas texture.

        Parameters
        ----------
        tile_index : int
            Get the offset of this tile.

        Return
        ------
        Tuple[int, int]
            The (X, Y) offset of this tile in texels.
        """
        width_tiles = self.shape_in_tiles[1]
        row = int(tile_index / width_tiles)
        col = tile_index % width_tiles

        # Return as (X, Y).
        return col * self.spec.width, row * self.spec.height

    def _calc_tex_coords(self, tile_index: int) -> np.ndarray:
        """Return the texture coordinates for this tile.

        This is only called from __init__ when we pre-compute the
        texture coordinates for every tiles.

        Parameters
        ----------
        tile_index : int
            Return coordinates for this tile.

        Return
        ------
        np.ndarray
            A (6, 2) array of texture coordinates.
        """
        offset = self._offset(tile_index)
        pos = offset / self.texture_shape[:2]
        shape = self.spec.shape[:2] / self.texture_shape[:2]

        quad = _QUAD.copy()
        quad[:, :2] *= shape
        quad[:, :2] += pos

        return quad

    def add_tile(self, data: np.ndarray) -> Optional[AtlasTile]:
        """Add one tile to the atlas.

        Parameters
        ----------
        data : np.ndarray
            The image data for this one tile.
        """
        if not self.spec.is_compatible(data):
            raise ValueError(
                f"Adding tile with shape {data.shape} does not match TextureAtlas2D "
                f"configured tile shape {self.tile_shape}"
            )

        try:
            tile_index = self._free_indices.pop()
        except KeyError:
            return None  # No available texture slots.

        # Upload the texture data for this one tile.
        self._set_tile_data(tile_index, data)

        # Return AtlasTile. The caller will need the texture coordinates to
        # render quads using our tiles.
        tex_coords = self._tex_coords[tile_index]
        return AtlasTile(tile_index, tex_coords)

    def remove_tile(self, tile_index: int) -> None:
        """Remove a tile from the texture atlas.

        Parameters
        ----------
        tile_index : int
            The index of the tile to remove.
        """
        self._free_indices.add(tile_index)  # This tile_index is now available.

        if self.MARK_DELETED_TILES:
            self._set_tile_data(tile_index, self.deleted_tile_data)

    def _set_tile_data(self, tile_index: int, data) -> None:
        """Set the texture data for this one tile.

        Parameters
        ----------
        tile_index : int
            The index of the tile to set.
        data
            The new texture data for the tile.
        """
        # Covert (X, Y) offset of this tile within the larger texture
        # in the the (row, col) that Texture2D expects.
        offset = self._offset(tile_index)[::-1]

        # Texture2D.set_data() will use glTexSubImage2D() under the hood to
        # only write into the tile's portion of the larger texture. This is
        # the main reason adding tiles to TextureAtlas2D is fast, we do not
        # have to re-upload the whole texture each time.
        self.set_data(data, offset=offset, copy=True)
