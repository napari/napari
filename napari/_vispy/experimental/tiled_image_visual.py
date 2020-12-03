"""TiledImageVisual class

A visual that draws tiles using a texture atlas.
"""
from typing import List, Set

import numpy as np

from ...layers.image.experimental import OctreeChunk
from ..vendored import ImageVisual
from ..vendored.image import _build_color_transform
from .texture_atlas import TextureAtlas2D
from .tile_set import TileSet

# Shape of she whole texture in tiles. Hardcode for now.
SHAPE_IN_TILES = (16, 16)


class TiledImageVisual(ImageVisual):
    """An image that is drawn using smaller tiles.

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
    the texture that's being updated.

    In addition, uploading new tiles does not cause the shader to be
    rebuilt. This is another reason TiledImageVisual is faster than
    creating a stand-alone ImageVisuals to draw each tile.

    Finally, rendering the tiles is efficient. TiledImageVisual renders by
    drawing one single list of quads. The texture coordinates of the quads
    point to the various tiles in the texture atlas. If all the tiles are
    stored in the same large texture, there will be zero texture swaps,
    which are expensive.

    Parameters
    ----------
    tile_shape : np.ndarray
        The shape of one tile like (256, 256, 3).
    """

    def __init__(self, tile_shape: np.ndarray, *args, **kwargs):
        self.tile_shape = tile_shape

        self._tiles = TileSet()  # The tiles we are drawing.

        self._clim = np.array([0, 1])  # TOOD_OCTREE: need to support clim

        # Initialize our parent ImageVisual.
        super().__init__(*args, **kwargs)

        # Must create the texture atlas after calling __init__ so
        # the attribute self._interpolation exists.
        self.unfreeze()
        self._texture_atlas = self._create_texture_atlas(tile_shape)
        self.freeze()

    def _create_texture_atlas(self, tile_shape: np.ndarray) -> TextureAtlas2D:
        """Create texture atlas up front or if we change texture shape.

        Attributes
        ----------
        tile_shape : np.ndarray
            The shape of our tiles such as (256, 256, 4).

        Return
        ------
        TextureAtlas2D
            The newly created texture atlas.
        """
        interp = 'linear' if self._interpolation == 'bilinear' else 'nearest'
        return TextureAtlas2D(tile_shape, SHAPE_IN_TILES, interpolation=interp)

    def set_data(self, image) -> None:
        """Set data of the ImageVisual.

        VispyImageLayer._on_display_change calls this with an empty image, but
        we can just ignore it. When created we are "empty" by virtue of not
        drawing any tiles yet.
        """

    def set_tile_shape(self, tile_shape: np.ndarray) -> None:
        """Set the shape of our tiles.

        All tiles are the same shape in terms of texels. However they might
        be drawn different physical sizes. For example drawing a single
        view into a quadtree might end up drawing some tiles 2X or 4X
        bigger than others. Typically you want to draw the "best available"
        data which might be on a different level.

        Parameters
        ----------
        tile_shape : np.ndarray
            Our tiles shape like (256, 256, 4)
        """

        # Clear all our previous tile information and set the new shape.
        self._tiles.clear()
        self.tile_shape = tile_shape

        # Create the new atlas and tell the shader about it.
        self._texture_atlas = self._create_texture_atlas(tile_shape)
        self._data_lookup_fn['texture'] = self._texture_atlas

    @property
    def size(self):
        # TODO_OCTREE: need to compute the size...
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
    def octree_chunk(self) -> List[OctreeChunk]:
        """Return data for the chunks we are drawing.

        List[OctreeChunk]
            The data for the chunks we are drawing.
        """
        return self._tiles.chunks

    def add_chunks(self, chunks: List[OctreeChunk]):
        """Any any chunks that we are not already drawing.

        Parameters
        ----------
        chunks : List[OctreeChunk]
            Add any of these we are not already drawing.
        """
        for octree_chunk in chunks:
            if not self._tiles.contains_octree_chunk(octree_chunk):
                self.add_one_tile(octree_chunk)

    def add_one_tile(self, octree_chunk: OctreeChunk) -> None:
        """Add one tile to the tiled image.

        Parameters
        ----------
        octree_chunk : OctreeChunk
            The data for the tile we are adding.

        Return
        ------
        int
            The tile's index.
        """

        atlas_tile = self._texture_atlas.add_tile(octree_chunk)

        if atlas_tile is None:
            return  # No slot available in the atlas.

        self._tiles.add(octree_chunk, atlas_tile)
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
            raise RuntimeError(f"Tile index {tile_index} not found.")

    def prune_tiles(self, visible_set: Set[OctreeChunk]) -> None:
        """Remove tiles that are not part of the given visible set.

        visible_set : Set[OctreeChunk]
            The set of currently visible chunks.
        """
        for tile_data in list(self._tiles.tile_data):
            if tile_data.octree_chunk.key not in visible_set:
                tile_index = tile_data.atlas_tile.index
                self.remove_tile(tile_index)

    def _build_vertex_data(self) -> None:
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
            tile = tile_data.atlas_tile
            verts = np.vstack((verts, tile.verts))
            tex_coords = np.vstack((tex_coords, tile.tex_coords))

        # Set the base ImageVisual _subdiv_ buffers
        self._subdiv_position.set_data(verts)
        self._subdiv_texcoord.set_data(tex_coords)
        self._need_vertex_update = False

    def _build_texture(self) -> None:
        """Override of ImageVisual._build_texture().

        TODO_OCTREE: This needs work. Need to do the clim stuff in in the
        base ImageVisual._build_texture but do it for each tile?
        """
        self._clim = np.array([0, 1])

        self._texture_limits = np.array([0, 1])  # hardcode
        self._need_colortransform_update = True

        self._need_texture_upload = False

    def _prepare_draw(self, view) -> None:
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
            grayscale = len(self.tile_shape) == 2 or self.tile_shape[2] == 1
            self.shared_program.frag[
                'color_transform'
            ] = _build_color_transform(
                grayscale, self.clim_normalized, self.gamma, self.cmap
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
