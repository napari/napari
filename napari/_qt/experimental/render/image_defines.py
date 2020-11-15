"""ImageConfig named tuple.
"""
from typing import NamedTuple, Tuple


class ImageConfig(NamedTuple):
    """Configuration for a tiled image."""

    image_shape: Tuple[int, int]
    tile_size: int
