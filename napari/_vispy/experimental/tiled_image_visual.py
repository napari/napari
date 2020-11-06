"""TiledImageVisual class
"""
from .._vendored import ImageVisual


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
    bytes in the tile being updated are sent to the card. The Vispy
    BaseTexture.set_data() method has an "offset" argument. When setting
    data with an offset Vispy calls glTexSubImage2D() to only write into
    impacted sub-region inside the larger texture.

    In addition, uploading tiles does not cause the shader to rebuilt. This
    is another reason TiledImageVisual is faster than creating a stand-alone
    ImageVisuals to draw each tile.

    Finally, rendering the tiles is also efficient. In one draw pass
    TiledImageVisual can render all the tiles. If all the tiles are stored
    in the same large texture, there will be zero texture swaps.
    """

    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)
