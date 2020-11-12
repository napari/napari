"""TiledImageVisual class
"""
from typing import List, Set

import numpy as np
from vispy.gloo.buffer import VertexBuffer
from vispy.visuals.shaders import Function, FunctionChain

from ...layers.image.experimental.octree_util import ChunkData
from ..vendored import ImageVisual
from ..vendored.image import (
    _apply_clim,
    _apply_clim_float,
    _apply_gamma,
    _apply_gamma_float,
    _c2l,
    _null_color_transform,
)
from .texture_atlas import TexInfo, TextureAtlas2D

# Shape of she whole texture in tiles. Hardcode for now.
SHAPE_IN_TILES = (16, 16)


# Two triangles to cover a [0..1, 0..1] quad.
_QUAD = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]], dtype=np.float32,
)

DATA_2D = True  # temporary


# TODO_OCTREE: Slightly modified from ImageVisual._build_color_transform
# Hopefully we can use the real one soon.
def _build_color_transform(_data, clim, gamma, cmap):

    if DATA_2D:
        fclim = Function(_apply_clim_float)
        fgamma = Function(_apply_gamma_float)
        fun = FunctionChain(
            None, [Function(_c2l), fclim, fgamma, Function(cmap.glsl_map)]
        )
    else:
        fclim = Function(_apply_clim)
        fgamma = Function(_apply_gamma)
        fun = FunctionChain(
            None, [Function(_null_color_transform), fclim, fgamma]
        )
    fclim['clim'] = clim
    fgamma['gamma'] = gamma
    return fun


class TileData:
    """Data related to one tile we are displaying.

    Parameters
    ----------
    chunk_data : ChunkData
        The chunk data that produced this time.
    tex_info : TexInfo
        The texture information from our tiled visual.
    """

    def __init__(self, chunk_data: ChunkData, tex_info: TexInfo):
        self.chunk_data = chunk_data
        self.tex_info = tex_info


def _vert_quad(chunk_data: ChunkData):
    quad = _QUAD.copy()

    # TODO_OCTREE: store as np.array in ChunkData?
    scale = np.array(chunk_data.scale, dtype=np.float32)
    scaled_shape = chunk_data.data.shape[:2] * scale

    # Modify XY's into place
    quad[:, :2] *= scaled_shape
    quad[:, :2] += chunk_data.pos

    return quad


def _tex_quad(chunk_data: ChunkData):
    quad = _QUAD.copy()[:, :2]

    # TODO_OCTREE: store as np.array in ChunkData?
    scale = np.array(chunk_data.scale, dtype=np.float32)
    scaled_shape = chunk_data.data.shape[:2] * scale

    # Modify XY's into place
    quad[:, :2] *= scaled_shape
    quad[:, :2] += chunk_data.pos

    return quad


class TileSet:
    """The tiles we are drawing.

    With a fast set membership test for ChunkData.
    """

    def __init__(self):
        self._tiles = {}
        self._chunks = set()

    def __len__(self) -> int:
        """Return the number of tiles in the set.

        Return
        ------
        int
            The number of tiles in the set.
        """
        return len(self._tiles)

    def clear(self) -> None:
        self._tiles.clear()
        self._chunks.clear()

    def add(self, tile_data: TileData) -> None:
        """Add this TiledData to the set.

        Parameters
        ----------
        tile_data : TileData
            Add this to the set.
        """
        tile_index = tile_data.tex_info.tile_index
        self._tiles[tile_index] = tile_data
        self._chunks.add(tile_data.chunk_data.key)

    def remove(self, tile_index: int) -> None:
        """Remove the TileData at this index from the set.

        tile_index : int
            Remove the TileData at this index.
        """
        chunk_data = self._tiles[tile_index].chunk_data
        del self._tiles[tile_index]
        self._chunks.remove(chunk_data.key)

    @property
    def chunks(self) -> List[ChunkData]:
        """Return all the chunk data that we have.

        Return
        ------
        List[ChunkData]
            All the chunk data in the set.
        """
        return [tile_data.chunk_data for tile_data in self._tiles.values()]

    @property
    def tile_data(self) -> List[TileData]:
        """Return all the tile data in the set.

        Return
        ------
        List[TileData]
            All the tile data in the set.
        """
        return self._tiles.values()

    def contains_chunk_data(self, chunk_data: ChunkData) -> bool:
        """Return True if the set contains this chunk data.

        Parameters
        ----------
        chunk_data : ChunkData
            Check if ChunkData is in the set.

        Return
        ------
        bool
            True if the set contains this chunk data.
        """
        return chunk_data.key in self._chunks


class TiledImageVisual(ImageVisual):
    """A larger image that's drawn using some number of smaller tiles.

    TiledImageVisual draws a single large image using a set of square image
    tiles. The size of the tiles is configurable, but 256x256 or 512x512
    might be good choices. All the tiles in one TiledImageVisual are the
    same size.

    The tiles are stored in larger textures as an "atlas". An atlas is
    basically just a texture which looks like a grid of smaller images. The
    grid has no borders between the tiles. The size of the larger textures
    is also configurable. For example a single 4096x4096 texture could
    store 256 different 256x256 tiles.

    Adding or removing tiles from a TiledImageVisual is efficient. Only the
    bytes in the tile(s) being updated are sent to the card. The Vispy
    method BaseTexture.set_data() has an "offset" argument. When setting
    texture data with an offset under the hood Vispy calls
    glTexSubImage2D(). It will only update the rectangular region within
    the texture that's being update.

    In addition, uploading new tiles does not cause the shader to be
    rebuilt. This is another reason TiledImageVisual is faster than
    creating a stand-alone ImageVisuals to draw each tile.

    Finally, rendering the tiles is also efficient. In one draw pass
    TiledImageVisual can render all the tiles. If all the tiles are stored
    in the same large texture, there will be zero texture swaps.
    """

    def __init__(self, tile_shape: np.ndarray, *args, **kwargs):
        self.tile_shape = tile_shape
        self._tiles = TileSet()
        self._verts = VertexBuffer()
        self._tex_coords = VertexBuffer()

        self._clim = np.array([0, 1])  # Constant for now.

        super().__init__(*args, **kwargs)

        self.unfreeze()
        self._texture_atlas = self._create_texture_atlas(tile_shape)
        self.freeze()

    def _create_texture_atlas(self, tile_shape: np.ndarray) -> TextureAtlas2D:
        if self._interpolation == 'bilinear':
            texture_interpolation = 'linear'
        else:
            texture_interpolation = 'nearest'

        return TextureAtlas2D(
            tile_shape, SHAPE_IN_TILES, interpolation=texture_interpolation
        )

    def set_data(self, image):
        pass

    def set_tile_shape(self, tile_shape: np.ndarray):

        # Set the new shape and clear all our previous tile information.
        self.tile_shape = tile_shape
        self._tiles.clear()

        # Create the new atlas and tell the shader about it.
        self._texture_atlas = self._create_texture_atlas(tile_shape)
        self._data_lookup_fn['texture'] = self._texture_atlas

    @property
    def size(self):
        # TODO_OCTREE:
        #
        # ImageVisual.size() does
        #     return self._data.shape[:2][::-1]
        #
        # We don't have a self._data so what do we put here? Maybe need
        # a bounds for all the currently visible tiles?
        # return self._texture_atlas.texture_shape[:2]
        return (1024, 1024)

    @property
    def num_tiles(self) -> int:
        """Return the number tiles currently being drawn.

        Return
        ------
        int
            The number of tiles currently being drawn.
        """
        return self._texture_atlas.num_slots_used

    @property
    def chunk_data(self) -> List[ChunkData]:
        """Return data for the chunks we are drawing.

        List[ChunkData]
            The data for the chunks we are drawing.
        """
        return self._tiles.chunks

    def add_chunks(self, chunks: List[ChunkData]):
        """Any any chunks that we are not already drawing.

        Parameters
        ----------
        chunks : List[ChunkData]
            Add any of these we are not already drawing.
        """
        for chunk_data in chunks:
            if not self._tiles.contains_chunk_data(chunk_data):
                self.add_one_tile(chunk_data)

    def add_one_tile(self, chunk_data: ChunkData) -> None:
        """Add one tile to the tiled image.

        Parameters
        ----------
        chunk_data : ChunkData
            The data for the tile we are adding.

        Return
        ------
        int
            The tile's index.
        """

        tex_info = self._texture_atlas.add_tile(chunk_data.data)

        if tex_info is None:
            return  # No slot available in the atlas.

        self._tiles.add(TileData(chunk_data, tex_info))
        self._need_vertex_update = True

    def remove_tile(self, tile_index: int) -> None:
        """Remove one tile from the image.

        Parameters
        ----------
        tile_index : int
            The tile to remove.
        """
        try:
            self._tiles.remove(tile_index)
            self._texture_atlas.remove_tile(tile_index)
            self._need_vertex_update = True
        except IndexError:
            # TODO_OCTREE: for now just raise
            raise RuntimeError(f"Tile index {tile_index} not found.")

    def prune_tiles(self, visible_set: Set[ChunkData]):
        """Remove tiles that are not part of the given visible set.

        visible_set : Set[ChunkData]
            The set of currently visible chunks.
        """
        for tile_data in list(self._tiles.tile_data):
            if tile_data.chunk_data.key not in visible_set:
                tile_index = tile_data.tex_info.tile_index
                self.remove_tile(tile_index)

    def _build_vertex_data(self):
        """Build vertex and texture coordinate buffers.

        This overrides ImageVisual._build_vertex_data(), it is called from
        our _prepare_draw().

        This is the heart of tiled rendering. Instead of drawing one quad
        with one texture, we draw one quad per tile. And for each quad its
        texture coordinates will pull from the right slot in the atlas.

        So as the card draws the tiles, where it's sampling from the
        texture will hop around in the atlas texture.
        """
        if len(self._tiles) == 0:
            return  # Nothing to draw.

        verts = np.zeros((0, 2), dtype=np.float32)
        tex_coords = np.zeros((0, 2), dtype=np.float32)

        # TODO_OCTREE: We can probably avoid vstack here if clever,
        # maybe one one vertex buffer sized according to the max
        # number of tiles we expect. But grow if needed.
        for tile_data in self._tiles.tile_data:
            chunk_data = tile_data.chunk_data

            vert_quad = _vert_quad(chunk_data)
            verts = np.vstack((verts, vert_quad))

            tex_quad = tile_data.tex_info.tex_coord
            tex_coords = np.vstack((tex_coords, tex_quad))

        # Set the base ImageVisual _subdiv_ buffers
        self._subdiv_position.set_data(verts)
        self._subdiv_texcoord.set_data(tex_coords)
        self._need_vertex_update = False

    def _build_texture(self):
        """Override of ImageVisual._build_texture().

        TODO_OCTREE: This needs work. Need to do the clim stuff in in the
        base ImageVisual._build_texture but do it for each tile?
        """
        self._clim = np.array([0, 1])

        self._texture_limits = np.array([0, 1])  # hardcode
        self._need_colortransform_update = True

        data = np.empty((1024, 1024, 3), dtype=np.uint8)
        data[:] = (1, 0, 0)  # handle RGB or RGBA?

        self._texture.set_data(data)
        self._need_texture_upload = False

    def _prepare_draw(self, view):
        """Override of ImageVisual._prepare_draw()

        TODO_OCTREE: See how much this changes from base class, if we can
        avoid too much duplication. Or factor out some common methods.
        """
        if self._need_interpolation_update:
            # Call the base ImageVisual._build_interpolation()
            self._build_interpolation()

            # But override to use our texture atlas.
            self._data_lookup_fn['texture'] = self._texture_atlas

        # We call our own _build_texture
        if self._need_texture_upload:
            self._build_texture()

        # TODO_OCTREE: how does colortransform change for tiled?
        if self._need_colortransform_update:
            prg = view.view_program
            self.shared_program.frag[
                'color_transform'
            ] = _build_color_transform(
                self._data, self.clim_normalized, self.gamma, self.cmap
            )
            self._need_colortransform_update = False
            prg['texture2D_LUT'] = (
                self.cmap.texture_lut()
                if (hasattr(self.cmap, 'texture_lut'))
                else None
            )

        # We call our own _build_vertex_data()
        if self._need_vertex_update:
            self._build_vertex_data()

        # Call the normal ImageVisual._update_method() unchanged.
        if view._need_method_update:
            self._update_method(view)
