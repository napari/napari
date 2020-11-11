"""TextureAtlas2D class.
"""
from typing import Tuple

import numpy as np
from vispy.gloo import Texture2D

# Two triangles to cover a [0..1, 0..1] quad.
_QUAD = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]], dtype=np.float32,
)


class TexInfo:
    """Texture Information

    Parameters
    ----------
    tile_index : int
        The tile's index in the atlas.
    tex_coord
        The texture coordinates of the tile.
    """

    def __init__(self, tile_index, tex_coord):
        self.tile_index = tile_index
        self.tex_coord = tex_coord


class TextureAtlas2D(Texture2D):
    """Two-dimension texture atlas.

    A single large texture with "slots" for smaller texture tiles.

    Parameters
    ----------
    tile_shape : Tuple[int, int]
        The (height, width) of one tile in texels.
    texture_shape : Tuple[int, int]
        The (height, width) of the full texture in terms of tiles.
    """

    # If True we mark removed tiles red for debugging. When we remove a
    # tile we do not need to waste bandwidth clearing the old texture data.
    # We just let that slot get written to later on. However for debugging
    # we can write into that spot to mark it as empty.
    MARK_DELETED_TILES = True

    def __init__(
        self,
        tile_shape: Tuple[int, int, int],
        shape_in_tiles: Tuple[int, int],
        **kwargs,
    ):
        # Each tile's shape in texels, for example 256x256.
        self.tile_shape = tile_shape

        # The full texture's shape in tiles, for example 4x4.
        self.shape_in_tiles = shape_in_tiles

        # The full texture's chape in texels, for example 1024x1024.
        height = self.tile_shape[0] * self.shape_in_tiles[0]
        width = self.tile_shape[1] * self.shape_in_tiles[1]

        # TODO_OCTREE: Hardcode RGB for now
        self.texture_shape = np.array([width, height, 3], dtype=np.int32)

        # Total number of texture slots in the atlas.
        self.num_slots_total = shape_in_tiles[0] * shape_in_tiles[1]

        # Every index is free to start.
        self._free_indices = set(range(0, self.num_slots_total))

        if self.MARK_DELETED_TILES:
            self.deleted_tile_data = np.empty(self.tile_shape, dtype=np.uint8)
            self.deleted_tile_data[:] = (1, 0, 0)  # handle RGB or RGBA?

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
        """Return the (X, Y) offset into the atlas texture.

        Parameters
        ----------
        tile_index : int
            Get the offset of this tile.

        Return
        ------
        Tuple[int, int]
            The (X, Y) offset of this tile in texels.
        """
        height_tiles, width_tiles = self.shape_in_tiles
        row = int(tile_index / width_tiles)
        col = tile_index % width_tiles
        return col * self.tile_shape[1], row * self.tile_shape[0]

    def _tex_coords(self, tile_index: int) -> np.ndarray:

        offset = self._offset(tile_index)
        pos = offset / self.texture_shape[:2]
        shape = self.tile_shape[:2] / self.texture_shape[:2]

        quad = _QUAD.copy()
        quad[:, :2] *= shape
        quad[:, :2] += pos

        return quad

    def add_tile(self, data: np.ndarray) -> TexInfo:
        """Add one tile to the atlas.

        Parameters
        ----------
        data : np.ndarray
        """
        if data.shape != self.tile_shape:
            raise ValueError(
                f"Tile shape {data.shape} does not match TextureAtlas2D "
                f"tile size {self.tile_shape}"
            )

        try:
            tile_index = self._free_indices.pop()
        except KeyError:
            # TODO_OCTREE: just raise something for now
            raise RuntimeError(
                f"All {self.num_slots_total} TextureAtlas2D slots are full"
            )

        self._set_tile_data(tile_index, data)
        tex_coords = self._tex_coords(tile_index)

        return TexInfo(tile_index, tex_coords)

    def remove_tile(self, tile_index: int) -> None:
        """Remove a tile from the texture atlas.

        Parameters
        ----------
        tile_index : int
            The index of the tile to remove.
        """
        # The index is now a free slow.
        self._free_indices.add(tile_index)

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

        print(f"_set_tile_data: {tile_index} -> {offset}")

        # Texture2D.set_data() will use glTexSubImage2D() under the hood to
        # only write into the tile's portion of the larger texture. This is
        # the main reason adding tiles to TextureAtlas2D is fast, we do not
        # have to re-upload the whole texture each time.
        self.set_data(data, offset=offset, copy=True)
