"""TiledImageVisual class
"""
from .._vendored import ImageVisual


class TiledImageVisual(ImageVisual):
    """Extend ImageVisual with a texture atlas of image tiles.

    TiledImageVisual represents a single image that has been broken down
    spatially into square tiles. The size of the tiles is configurable,
    but for example we might use 256x256 pixel tiles.

    The tiles are stored in large textures, also configurable. For example
    a 4096x4096 texture could store 256 different 256x256 tiles.

    Adding or removing tiles from a TiledImageVisual is meant to be
    efficient. You might add a number of different tiles in one frame. And
    then drawing the full image using tiles is also supposed to be
    efficient, since we minimize the number of texture swaps.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
