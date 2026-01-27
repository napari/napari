import itertools
from typing import Any

import numpy as np
from vispy.scene.visuals import Compound, Image
from vispy.visuals import BaseVisual
from vispy.visuals.transforms.linear import STTransform

# attributes that are passed from the tiled node to the child image nodes.
PASS_THROUGH_ATTRIBUTES = {
    'cmap',
    'clim',
    'opacity',
    'gamma',
    'interpolation',
    'custom_kernel',
}


class TiledImageNode(Compound):
    """Custom Vispy scenegraph Node to display large images.

    Vispy 2D rendering works by drawing images as 2D OpenGL textures.
    OpenGL has a texture size limit (driver and hardware dependent), which
    means that some images are too large to be displayed by a single
    texture. This class automatically splits those images into tiles not
    exceeding the texture size, and displays each tile as a texture, offset
    by the appropriate amount in visual space.

    Attributes such as colormap and contrast limits are passed through to
    the child nodes using setattr.

    Attributes
    ----------
    data : np.ndarray
        The data to be displayed.
    texture_format : str
        The texture format (uint8, uint16, float32).
    tile_size : int
        The maximum tile size, used to split the image.
    adopted_children : list[Image]
        The child Image nodes containing the component tiles of the image.
        Note: we use the term "adopted children" because "children" is a
        protected attribute of the Node class, and contains other VisualNodes.
    offsets : list[tuple[int, int]]
        The x/y offset of each tile (in the same order as `adopted_children`).
    """

    def __init__(
        self,
        data: np.ndarray,
        tile_size: int,
        texture_format: str | None = None,
    ) -> None:
        self.unfreeze()
        self.texture_format = texture_format
        self.adopted_children: list[Image] = []
        self.offsets: list[tuple[int, int]] = []
        self.tile_size = tile_size
        self.data = None
        super().__init__([])
        self.set_data(data)

    def add_subvisual(self, visual):
        self._subvisuals.append(visual)

    def _transform_changed(self, event=None):
        BaseVisual._transform_changed(self)

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
            self._subvisuals = []
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
                self.add_subvisual(ch)

    def set_gl_state(self, *args: Any, **kwargs: Any) -> None:
        for child in self.adopted_children:
            child.set_gl_state(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name in PASS_THROUGH_ATTRIBUTES | {'events'}:
            if len(self.adopted_children) > 0:
                return getattr(self.adopted_children[0], name)
            return None
        return self.__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in PASS_THROUGH_ATTRIBUTES:
            for child in self.adopted_children:
                setattr(child, name, value)
        else:
            super().__setattr__(name, value)


def make_tiles(
    image: np.ndarray, tile_size: int
) -> list[tuple[tuple[int, int], np.ndarray]]:
    """Split a large image into a list of tiles and offsets.

    Supports input shapes (H, W) for grayscale or (H, W, 3) for RGB.

    Parameters
    ----------
    image : np.ndarray, shape (H, W[, 3])
        The input image.
    tile_size : int
        The maximum size of any tile along any axis.

    Returns
    -------
    tile_list : list[tuple[tuple[int, int], np.ndarray]]
        List of x, y offsets and corresponding array tiles.
    """
    h, w, *_ = image.shape

    tile_list = [
        # note: vispy space is transposed (y, x -> x, y) compared to NumPy
        # indexing space; we want the vispy offsets so we swap them here.
        ((x, y), image[y : y + tile_size, x : x + tile_size])
        for y, x in itertools.product(
            range(0, h, tile_size), range(0, w, tile_size)
        )
    ]

    return tile_list
