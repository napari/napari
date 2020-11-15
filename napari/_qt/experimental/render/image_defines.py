"""ImageConfig named tuple.
"""
from typing import NamedTuple, Tuple


class ImageConfig(NamedTuple):
    image_shape: Tuple[int, int]
    tile_size: int
