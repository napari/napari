"""TextureAtlas2D class.

A texture atlas is a large texture that stores many smaller tile textures.
"""
from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np
from vispy.gloo import Texture2D

from ...layers.image.experimental import OctreeChunk
from ...types import ArrayLike
from ...utils.translations import trans
from ..utils.gl import fix_data_dtype

# Two triangles which cover a [0..1, 0..1] quad.
_QUAD = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]],
    dtype=np.float32,
)


def _quad(size: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Return one quad with the given size and position.

    Parameters
    ----------
    size : np.ndarray
        Size of the quad (X, Y).
    pos : np.ndarray
        Position of the quad (X, Y)
    """
    quad = _QUAD.copy()  # Copy and modify in place
    quad[:, :2] *= size
    quad[:, :2] += pos

    return quad


def _chunk_verts(octree_chunk: OctreeChunk) -> np.ndarray:
    """Return a quad for the vertex buffer.

    Parameters
    ----------
    octree_chunk : OctreeChunk
        Create a quad for this chunk.

    Returns
    -------
    np.darray
        The quad vertices.
    """
    geom = octree_chunk.geom
    return _quad(geom.size, geom.pos)


class AtlasTile(NamedTuple):
    """Information about one specific tile in the atlas.

    AtlasTile is returned from TextureAtlas2D.add_tile() so the caller has
    the verts and texture coordinates to render each tile in the atlas.

    Attributes
    ----------
    index : int
        The index of this tile in the atlas.
    verts : np.ndarray
        The vertices of this tile.
    tex_coords : np.ndarray
        The texture coordinates of this tile.
    """

    index: int
    verts: np.ndarray
    tex_coords: np.ndarray


class TileSpec(NamedTuple):
    """Specification for the tiles in the atlas.

    Parameters
    ----------
    shape : np.darray
        The shape of single tile.
    ndim : int
        The number of dimension in the shape of the tile.
    height : int
        The height of a tile.
    width : int
        The width of a tile.
    depth : int
        The depth of a tile, 1 for grayscale or RGB/RGBA.
    """

    shape: np.ndarray
    ndim: int
    height: int
    width: int
    depth: int

    @classmethod
    def from_shape(cls, shape: np.ndarray):
        """Create a TileSpec from just the shape.

        Parameters
        ----------
        shape : np.darray
            Create a TileSpec based on this shape.
        """
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
    shape_in_tiles : Tuple[int, int]
        The (height, width) of the full texture in terms of tiles.
    image_converter : Callable[[ArrayLike], ArrayLike]
        For converting raw to displayed data.
    """

    def __init__(
        self,
        tile_shape: tuple,
        shape_in_tiles: Tuple[int, int],
        image_converter: Callable[[ArrayLike], ArrayLike],
        **kwargs,
    ):
        # Each tile's shape in texels, for example (256, 256, 3).
        self.spec = TileSpec.from_shape(tile_shape)

        # The full texture's shape in tiles, for example (4, 4).
        self.shape_in_tiles = shape_in_tiles

        # The full texture's shape in texels, for example (1024, 1024).
        height = self.spec.height * self.shape_in_tiles[0]
        width = self.spec.width * self.shape_in_tiles[1]
        self.full_shape = np.array(
            [width, height, self.spec.depth], dtype=np.int32
        )

        # Total number of texture slots in the atlas.
        self.num_slots_total = shape_in_tiles[0] * shape_in_tiles[1]

        # Every index is free initially.
        self._free_indices = set(range(0, self.num_slots_total))

        # Pre-compute the texture coords for every tile. Otherwise we'd be
        # calculating these over and over as tiles are added. These are for
        # full tiles only. Edge and corner tiles will need custom texture
        # coordinates based on their size.
        tile_shape = self.spec.shape  # Use the shape of a full tile.
        self._tex_coords = [
            self._calc_tex_coords(tile_index, tile_shape)
            for tile_index in range(self.num_slots_total)
        ]

        # Store an image converter that will convert from raw to displayed image
        self.image_converter = image_converter

        super().__init__(shape=tuple(self.full_shape), **kwargs)

    @property
    def num_slots_free(self) -> int:
        """The number of available texture slots.

        Returns
        -------
        int
            The number of available texture slots.
        """
        return len(self._free_indices)

    @property
    def num_slots_used(self) -> int:
        """The number of texture slots currently in use.

        Returns
        -------
        int
            The number of texture slots currently in use.
        """
        return self.num_slots_total - self.num_slots_free

    def _offset(self, tile_index: int) -> Tuple[int, int]:
        """Return the (row, col) offset into the full atlas texture.

        Parameters
        ----------
        tile_index : int
            Return the offset of this tile.

        Returns
        -------
        Tuple[int, int]
            The (row, col) offset of this tile in texels.
        """
        width_tiles = self.shape_in_tiles[1]
        row = int(tile_index / width_tiles)
        col = tile_index % width_tiles

        return row * self.spec.height, col * self.spec.width

    def _calc_tex_coords(
        self,
        tile_index: int,
        tile_shape: np.ndarray,
    ) -> np.ndarray:
        """Return the texture coordinates for this tile.

        This is only called from __init__, so we only calculate the
        texture coordinates once up front.

        Parameters
        ----------
        tile_index : int
            Return coordinates for this tile.
        tile_shape : np.ndarray
            The shape of a single tile.

        Returns
        -------
        np.ndarray
            A (6, 2) array of texture coordinates.
        """
        offset = self._offset(tile_index)
        pos = offset / self.full_shape[:2]
        shape = tile_shape[:2] / self.full_shape[:2]

        pos = pos[::-1]
        shape = shape[::-1]

        return _quad(shape, pos)

    def add_tile(
        self, octree_chunk: OctreeChunk, clim=None
    ) -> Optional[AtlasTile]:
        """Add one tile to the atlas.

        Parameters
        ----------
        octree_chunk : np.ndarray
            The image data for this one tile.
        clim : tuple, optional
            Contrast limits to normalize by if provided.

        Returns
        -------
        Optional[AtlasTile]
            The AtlasTile if the tile was successfully added.
        """
        data = octree_chunk.data

        # Transform data from raw to displayed
        # Ideally this should be removed and all transforming
        # should happen on GPU
        data = self.image_converter(data)

        # normalize by contrast limits if provided. This normalization
        # will not be required after https://github.com/vispy/vispy/pull/1920/
        # and at that point should be changed.
        if clim is not None and (data.ndim == 2 or data.shape[2] == 1):
            clim = np.asarray(clim, dtype=np.float32)
            data = data - clim[0]  # not inplace so we don't modify orig data
            if clim[1] - clim[0] > 0:
                data /= clim[1] - clim[0]
            else:
                data[:] = 1 if data[0, 0] != 0 else 0

        assert isinstance(data, np.ndarray)

        # Make sure data is of a dtype acceptable to vispy
        data = fix_data_dtype(data)

        if not self.spec.is_compatible(data):
            # It will be not compatible of number of dimensions or depth
            # are wrong. Or if the data is too big to fit in one tile.
            raise ValueError(
                trans._(
                    "Data with shape {shape} is not compatible with this TextureAtlas2D which has tile shape {spec_shape}",
                    deferred=True,
                    shape=octree_chunk.data.shape,
                    spec_shape=self.spec.shape,
                )
            )

        try:
            tile_index = self._free_indices.pop()
        except KeyError:
            return None  # No available texture slots.

        # Upload the texture data for this one tile.
        self._set_tile_data(tile_index, data)

        # Return AtlasTile. The caller will need the texture coordinates to
        # render quads using our tiles.
        verts = _chunk_verts(octree_chunk)
        tex_coords = self._get_tex_coords(tile_index, data)
        return AtlasTile(tile_index, verts, tex_coords)

    def _get_tex_coords(self, tile_index: int, data: np.ndarray) -> np.ndarray:
        """Return the texture coordinates for this tile.

        Parameters
        ----------
        tile_index : int
            The index of this tile.
        data : np.ndarray
            The image data for this tile.

        Returns
        -------
        np.ndarray
            The texture coordinates for the tile.
        """
        # If it's the exact size of our tiles. Return the pre-computed
        # texture coordinates for this tile. Fast!
        if self.spec.shape == data.shape:
            return self._tex_coords[tile_index]

        # Edge or corner tile, compute exact coords.
        return self._calc_tex_coords(tile_index, data.shape)

    def remove_tile(self, tile_index: int) -> None:
        """Remove a tile from the texture atlas.

        Parameters
        ----------
        tile_index : int
            The index of the tile to remove.
        """
        self._free_indices.add(tile_index)  # This tile_index is now available.

    def _set_tile_data(self, tile_index: int, data: np.ndarray) -> None:
        """Upload the texture data for this one tile.

        Note the data might be smaller than a full tile slot. If so, we
        don't really care what texels are in the rest of the tile's slot.
        They will not be drawn because we'll use the correct texture
        coordinates for the tiles actual size.

        Parameters
        ----------
        tile_index : int
            The index of the tile to upload.
        data
            The texture data for the tile.
        """
        # The texel offset of this tile within the larger texture.
        offset = self._offset(tile_index)

        # Texture2D.set_data() will use glTexSubImage2D() under the hood to
        # only write into the tile's portion of the larger texture. This is
        # a big reason adding tiles to TextureAtlas2D is fast.
        self.set_data(data, offset=offset, copy=True)
