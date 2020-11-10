"""TextureAtlas2D class.
"""
from typing import Tuple

import numpy as np
from vispy.gloo import Texture2D

# Two triangles to cover a [0..1, 0..1] quad.
_QUAD = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0]],
    dtype=np.float32,
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

    # Mark removed tiles red for debugging. It's not necessary to modify
    # the removed tile's texture data. It will just get overwritten when
    # some future tile is added. But for debugging we can red it out, so
    # that if we mistakenly draw it, we'll know.
    MARK_DELETED_TILES = True

    def __init__(
        self, tile_shape: Tuple[int, int], texture_shape_tiles: Tuple[int, int]
    ):
        super().__init__()

        # Each tile's shape in texels.
        self.tile_shape = tile_shape

        # The full texture's shape in terms of tiles.
        self.texture_shape_tiles = texture_shape_tiles

        # Total number of texture slots in the atlas.
        self.num_slots_total = texture_shape_tiles[0] * texture_shape_tiles[1]

        height = self.tile_shape[0] * self.texture_shape_tiles[0]
        width = self.tile_shape[1] * self.texture_shape_tiles[1]
        self.texture_shape_texels = np.array([width, height], dtype=np.int32)

        # Free tile indexes.
        self.free = set(range(0, self.num_slots_total + 1))

        if self.MARK_DELETED_TILES:
            self.deleted_tile_data = np.fill(
                self.tile_shape + (4,), (1, 0, 0, 1)
            )

    @property
    def num_slots_free(self) -> int:
        """Return the number of available texture slots.

        Return
        ------
        int
            The number of available texture slots.
        """
        return len(self.free)

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
        height_tiles, width_tiles = self.texture_shape_tiles
        row = int(tile_index / height_tiles)
        col = tile_index % width_tiles
        return row * self.tile_shape[0], col * self.tile_shape[1]

    def _tex_coords(self, tile_index: int) -> np.ndarray:

        offset = self._offset(tile_index)
        pos = offset / self.texture_shape_texels
        shape = self.tile_shape / self.texture_shape_texels

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

        assert data.shape == self.tile_shape

        try:
            tile_index = self.free.pop()
        except KeyError:
            # TODO_OCTREE: just raise something for now
            raise RuntimeError("Tile is wrong shape")

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
        self.free.add(tile_index)

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

        # Get (X, Y) offset of this tile within the larger texture.
        offset = self._offset(tile_index)

        # Call Texture2D.set_data() which will call glTexSubImage2D() under
        # the hood to only upload the data for this one tile.
        self.set_data(data, offset=offset, copy=True)
