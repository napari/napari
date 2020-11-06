"""TiledImageVisual class
"""
from .._vendored import ImageVisual


class TiledImageVisual(ImageVisual):
    """Extend ImageVisual with a texture atlas of image tiles.

    TiledImageVisual draws a set of square image tiles. The size of the
    tiles is configurable, but 256x256 or 512x512 might be good choices.
    All the tiles in one TiledImageVisual are the same size.

    The tiles are stored in larger textures as an "atlas", basically a grid
    of the small tiles with no borders between them. The size of the larger
    textures is also configurable. For example a single 4096x4096 texture
    could store 256 different 256x256 tiles.

    Adding or removing tiles from a TiledImageVisual is efficient. Only the
    bytes in the tile being updated are send to the card. The Vispy
    BaseTexture.set_data() method has an "offset" argument. When setting
    data with an offset Vispy calls glTexSubImage2D() to modify only the
    impacted sub-region inside the larger texture.

    Uploading tiles does not cause the shader to rebuilt. This is another
    reason TiledImageVisuals is faster than creating new ImageVisuals.

    Finally, rendering the tiles is also efficient. In one draw pass
    TiledImageVisual can render all the tiles. If all the tiles are stored
    in one large texture, there will be zero texture swaps.
    """

    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)
