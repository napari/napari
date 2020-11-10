"""TiledImageVisual class
"""
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from vispy.gloo.buffer import VertexBuffer

from ...layers.image.experimental.octree_util import ChunkData
from ..vendored import ImageVisual
from .texture_atlas import TexInfo, TextureAtlas2D

# Shape of she whole texture in tiles. Hardcode for now.
SHAPE_IN_TILES = (4, 4)


# Two triangles to cover a [0..1, 0..1] quad.
_QUAD = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 0],
    ],
    dtype=np.float32,
)

# Either 2d shape or 2d shape plus 3 or 4 colors.
ImageShape = Tuple[int, int, Optional[int]]


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

    def __init__(self, tile_shape: ImageShape, *args, **kwargs):
        self.tiles: Dict[int, TileData] = {}
        self._verts = VertexBuffer()
        self._tex_coords = VertexBuffer()
        self.texture_atlas = TextureAtlas2D(tile_shape, SHAPE_IN_TILES)

        # This freezes the class, so can't add attributes after this.
        super().__init__(*args, **kwargs)

    @property
    def num_tiles(self) -> int:
        """Return the number tiles currently being drawn.

        Return
        ------
        int
            The number of tiles currently being drawn.
        """
        return self.texture_atlas.num_slots_used

    @property
    def chunk_data(self) -> List[ChunkData]:
        """Return data for the chunks we are drawing.

        List[ChunkData]
            The data for the chunks we are drawing.
        """
        # TODO_OCTREE: return iterator instead?
        return [tile_data.chunk_data for tile_data in self.tiles.values()]

    def __contains__(self, chunk_data):
        return chunk_data.key in self.tiles

    def add_tile(self, chunk_data: ChunkData) -> int:
        """Add one tile to the image.

        Parameters
        ----------
        chunk_data : ChunkData
            The data for the tile we are adding.

        Return
        ------
        int
            The tile index.
        """
        tex_info = self.texture_atlas.add_tile(chunk_data.data)
        tile_index = tex_info.tile_index
        self.tiles[tile_index] = TileData(chunk_data, tex_info)

        self._compute_buffers()

        return tile_index

    def remove_tile(self, tile_index: int) -> None:
        """Remove one tile from the image.

        Parameters
        ----------
        tile_index : int
            The tile to remove.
        """
        try:
            del self.tiles[tile_index]
            self.texture_atlas.remove_tile(tile_index)
        except IndexError:
            # TODO_OCTREE: for now just raise
            raise RuntimeError(f"Tile index {tile_index} not found.")

    def prune_tiles(self, visible_set: Set[ChunkData]):
        """Remove tiles that are not part of the given visible set.

        visible_set : Set[ChunkData]
            The set of currently visible chunks.
        """
        for tile_data in list(self.tiles.values()):
            if tile_data.chunk_data not in visible_set:
                tile_index = tile_data.tex_info.tile_index
                self.remove_tile(tile_index)

    def _compute_buffers(self) -> None:
        """Create the vertex and texture coordinate buffers."""
        verts = np.zeros((0, 3), dtype=np.float32)
        tex_coords = np.zeros((0, 2), dtype=np.float32)

        # TODO_OCTREE: avoid vstack, create one buffer up front
        # for all the verts/tex_coords in all the tiles?
        for tile_data in self.tiles.values():
            chunk_data = tile_data.chunk_data

            vert_quad = _vert_quad(chunk_data)
            verts = np.vstack((verts, vert_quad))

            tex_quad = _tex_quad(chunk_data)
            tex_coords = np.vstack((tex_coords, tex_quad))

        self._verts.set_data(verts.astype('float32'))
        self._tex_coords.set_data(tex_coords.astype('float32'))
