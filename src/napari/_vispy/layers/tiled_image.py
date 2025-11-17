from typing import Any

import numpy as np
from vispy.scene.node import Node
from vispy.scene.visuals import Image
from vispy.visuals.transforms.linear import STTransform


class TiledImageNode(Node):
    def __init__(
        self,
        data: np.ndarray,
        tile_size: int,
        texture_format: str | None = None,
    ) -> None:
        self.texture_format = texture_format
        self.adopted_children: list[Image] = []
        self.tile_size = tile_size
        super().__init__()

        self.set_data(data)

    def set_data(self, data: np.ndarray) -> None:
        tiles = make_tiles(data, self.tile_size)
        self.offsets = [of for of, _ in tiles]
        self.data = data
        # if the correct number of tiles already exist, just update their data
        # otherwise, delete all existing tiles and create new ones
        if len(self.adopted_children) == len(tiles):
            for ch, (_, dat) in zip(self.adopted_children, tiles, strict=True):
                ch.set_data(dat)
        else:
            for child in self.adopted_children:
                child.parent = None
            self.adopted_children = [
                Image(
                    data=dat,
                    parent=self,
                    texture_format=self.texture_format,
                )
                for _, dat in tiles
            ]
            for ch, offset in zip(
                self.adopted_children, self.offsets, strict=True
            ):
                ch.transform = STTransform(translate=offset + (0,))

    def set_gl_state(self, *args: Any, **kwargs: Any) -> None:
        for child in self.adopted_children:
            child.set_gl_state(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if (
            name in ['cmap', 'clim', 'opacity', 'gamma', 'events']
            and len(self.adopted_children) > 0
        ):
            return getattr(self.adopted_children[0], name)
        return self.__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ['cmap', 'clim', 'opacity', 'gamma']:
            for child in self.adopted_children:
                setattr(child, name, value)
        else:
            super().__setattr__(name, value)


def make_tiles(
    image: np.ndarray, tile_size: int
) -> list[tuple[tuple[int, int], np.ndarray]]:
    """
    Splits a large image (grayscale or RGB) into tiles and creates Image visuals for each tile.
    Supports input shapes (H, W) for grayscale or (H, W, 3) for RGB.
    """
    if image.ndim == 2:
        h, w = image.shape
    elif image.ndim == 3 and image.shape[2] == 3:
        h, w, _ = image.shape

    # Calculate number of tiles needed
    tiles_y = int(np.ceil(h / tile_size))
    tiles_x = int(np.ceil(w / tile_size))

    # Create a separate Image visual for each tile
    tile_list = []
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            # Calculate tile position and size
            x = tx * tile_size
            y = ty * tile_size

            w_tile = min(tile_size, w - x)
            h_tile = min(tile_size, h - y)

            # Skip if tile is empty
            if w_tile <= 0 or h_tile <= 0:
                continue

            # Extract tile data
            tile_data = image[y : y + h_tile, x : x + w_tile]

            tile_list.append(((x, y), tile_data))

    return tile_list
