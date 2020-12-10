"""TiledImageVisual class

A visual that draws tiles using a texture atlas.

Ultimately TiledImageVisual cannot depend on OctreeChunk. And Octree
code should not depend on TiledImageVisual! So there really can be
no class or named tuple that gets passed between them.

Instead, we'll probably just have a function signature that takes things
like the pos, size and depth of each tile as separate arguments. But
for now both do depend on OctreeChunk.
"""
from typing import List, Set

import numpy as np

from napari.layers.image.experimental.octree_chunk import OctreeChunkKey

from ...layers.image.experimental import OctreeChunk
from ..vendored import ImageVisual
from ..vendored.image import _build_color_transform
from .texture_atlas import TextureAtlas2D
from .tile_set import TileSet

# Shape of she whole texture in tiles. Hardcode for now.
SHAPE_IN_TILES = (16, 16)


class TiledImageVisual(ImageVisual):
    """An image that is drawn using one or more chunks or tiles.

    A regular ImageVisual is a single image drawn as a single rectangle
    with a single texture. A tiled TiledImageVisual also has a single
    texture, but that texture is a TextureAtlas2D.

    A texture atlas is basically a single texture that contains smaller
    textures within it, like a quilt. In our cases the smaller textures are
    all the same size, for example (256, 256). For example a 4k x 4k
    texture can hold 256 different (256, 256) tiles.

    When the TiledImageVisual draws, it draws a single list of quads. Each
    quad's texture coordinates potentially refers to a different texture in
    the atlas.

    The quads can be located anywhere, even in 3D. TiledImageVisual does
    not know if it's drawing an octree or a grid, or just a scatter of tiles.
    A key point is while the texture tiles are all the same size, the quads
    can all be different sizes.

    For example, one quad might have a (256, 256) texture, but it's
    physically tiny on the screen. While the next quad is also showing a
    (256, 256) texture, it has to be, but that quad is really big on that
    same screen. This ability to have different size quads comes in handy
    for octree rendering, where we often draw chunks from multiple levels
    of the octree at the same time.

    Adding or removing tiles from a TiledImageVisual is efficient. Only the
    bytes in the the tile(s) being updated are sent to the card. The Vispy
    method BaseTexture.set_data() has an "offset" argument. When setting
    texture data with an offset under the hood Vispy calls
    glTexSubImage2D(). It will only update the rectangular region within
    the texture that's being updated. This is critical to making
    TiledImageVisual efficient.

    In addition, uploading new tiles does not cause the shader to be
    rebuilt. This is another reason TiledImageVisual is faster than
    creating a stand-alone ImageVisuals, where each new ImageVisual results
    in a shader build today. If that were fixed TiledImageVisual would
    still be faster, but the speed gap would be smaller.

    Parameters
    ----------
    tile_shape : np.ndarray
        The shape of one tile like (256, 256, 3).
    """

    def __init__(self, tile_shape: np.ndarray, *args, **kwargs):
        self.tile_shape = tile_shape

        self._tiles: TileSet = TileSet()  # The tiles we are drawing.

        self._clim = np.array([0, 1])  # TOOD_OCTREE: need to support clim

        # Initialize our parent ImageVisual.
        super().__init__(*args, **kwargs)

        # We must create the texture atlas *after* calling super().__init__
        # because super().__init__ creates self._interpolation which we
        # our _create_texture_atlas references.
        #
        # The unfreeze/freeze stuff is just a vispy thing to guard against
        # adding attributes after construction, which often leads to bugs,
        # so we have to toggle it off here. Not a big deal.
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
        # TODO_OCTREE: Who checks this? Need to compute the size...
        #
        # ImageVisual.size() does
        #     return self._data.shape[:2][::-1]
        #
        # We don't have a self._data so what do we put here? Maybe need
        # a bounds for all the currently drawable tiles?
        # return self._texture_atlas.texture_shape[:2]
        #
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
    def octree_chunks(self) -> List[OctreeChunk]:
        """Return data for the chunks we are drawing.

        List[OctreeChunk]
            The data for the chunks we are drawing.
        """
        return self._tiles.chunks

    def add_chunks(self, chunks: List[OctreeChunk]) -> int:
        """Any one or more chunks that we are not already drawing.

        Parameters
        ----------
        chunks : List[OctreeChunk]
            Chunks that we may or may not already be drawing.

        Return
        ------
        int
            The number of chunks that still need to be added.
        """
        # Get only the new chunks, the ones we are not currently drawing.
        new_chunks = [
            octree_chunk
            for octree_chunk in chunks
            if not self._tiles.contains_octree_chunk(octree_chunk)
        ]

        # Add one or more of the new chunks.
        while new_chunks:
            self.add_one_chunk(new_chunks.pop(0))  # Add the first one.

            # In the future we might add several chunks here. We want
            # to add as many as we can without tanking the framerate
            # too much.
            #
            # For now, we just add ONE chunk per frame. We've timed (256,
            # 256) pixel chunks taking a whopping 40ms to load into VRAM!
            # Probably due to CPU-side processing we are doing? So today
            # there really is only time to add one chunk.
            #
            # Even if adds were fast, adding just one is not horrible.
            # The frame rate will stay smooth. But adding more if they
            # fit within the budget would get the newer/better data
            # drawn faster.
            break

        # Return how many chunks we did NOT add. The system should continue
        # to poll and draw until we return 0.
        return len(new_chunks)

    def add_one_chunk(self, octree_chunk: OctreeChunk) -> None:
        """Add one chunk to the tiled image.

        Parameters
        ----------
        octree_chunk : OctreeChunk
            The chunk we are adding.

        Return
        ------
        int
            The newly added chunk's index.
        """
        # Add to the texture atlas.
        atlas_tile = self._texture_atlas.add_tile(octree_chunk)

        if atlas_tile is None:
            return  # No slot was available in the atlas. That's bad.

        # Add our mapping between chunks and atlas tiles.
        self._tiles.add(octree_chunk, atlas_tile)

        # Set this flag so we call self._build_vertex_data() the next time
        # we are drawn. It will create new vertex and texture coordinates
        # buffers to include this new chunk.
        self._need_vertex_update = True

    @property
    def chunk_set(self) -> Set[OctreeChunkKey]:
        """Return the set of chunks we are drawing.

        Return
        ------
        Set[OctreeChunkKey]
            The set of chunks we are drawing.
        """
        return self._tiles.chunk_set

    def prune_tiles(self, drawable_set: Set[OctreeChunk]) -> None:
        """Remove tiles that are not part of the drawable set.

        drawable_set : Set[OctreeChunk]
            The set of currently drawable chunks.
        """
        for tile_data in list(self._tiles.tile_data):
            if tile_data.octree_chunk.key not in drawable_set:
                # print(f"REMOVE: {tile_data.octree_chunk}")
                tile_index = tile_data.atlas_tile.index
                self._remove_tile(tile_index)

    def _remove_tile(self, tile_index: int) -> None:
        """Remove one tile from the tiled image.

        Parameters
        ----------
        tile_index : int
            The tile to remove.
        """
        try:
            self._tiles.remove(tile_index)
            self._texture_atlas.remove_tile(tile_index)

            # Must rebuild to remove this from what we are drawing.
            self._need_vertex_update = True
        except IndexError as exc:
            # Fatal error right now, but maybe in weird situation we should
            # ignore this error? Let's see when it happens.
            raise RuntimeError(f"Tile index {tile_index} not found.") from exc

    def _build_vertex_data(self) -> None:
        """Build vertex and texture coordinate buffers.

        This overrides ImageVisual._build_vertex_data(), it is called from
        our _prepare_draw().

        This is the heart of tiled rendering. Instead of drawing one quad
        with one texture, we draw one quad per tile. And for each quad we
        set its texture coordinates so that it will pull from the right
        slot in the atlas.

        As the card draws the tiles, the locations it samples from the
        texture will hop around in the atlas texture.

        Today we only have one atlas texture, but in the future we might
        have multiple atlas textures. If so, we'll want to sort the quads
        to minimize the number of texture swaps. Sample from different
        tiles in one atlas texture is fast, but switching texture is
        slower.
        """
        if len(self._tiles) == 0:
            return  # Nothing to draw.

        verts = np.zeros((0, 2), dtype=np.float32)
        tex_coords = np.zeros((0, 2), dtype=np.float32)

        # TODO_OCTREE: We can probably avoid vstack here? Maybe create one
        # vertex buffer sized according to the max number of tiles we
        # expect? But grow it if we exceed our guess?
        for tile_data in self._tiles.tile_data_sorted:
            atlas_tile = tile_data.atlas_tile
            verts = np.vstack((verts, atlas_tile.verts))
            tex_coords = np.vstack((tex_coords, atlas_tile.tex_coords))

        # Set the base ImageVisual's _subdiv_ buffers. ImageVisual has two
        # modes: imposter and subdivision. So far TiledImageVisual
        # implicitly is always in subdivision mode. Not sure if we'd ever
        # support imposter, or if that even makes sense with tiles?
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
