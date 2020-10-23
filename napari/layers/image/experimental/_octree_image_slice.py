"""OctreeImageSlice class.
"""
import logging
from typing import Callable

from ....types import ArrayLike
from .._image_slice import ImageSlice
from .octree import Octree

LOGGER = logging.getLogger("napari.async")


class OctreeImageSlice(ImageSlice):
    """Add Octree functionality to ImageSlice
    """

    def __init__(
        self,
        image: ArrayLike,
        image_converter: Callable[[ArrayLike], ArrayLike],
        rgb: bool,
        octree_level: int,
        data_corners,
    ):
        LOGGER.debug("OctreeImageSlice.__init__")
        super().__init__(image, image_converter, rgb)

        self._octree = None
        self._octree_level = octree_level
        self._data_corners = data_corners

    @property
    def num_octree_levels(self) -> int:
        """Return the number of levels in the octree."""
        return self._octree.num_levels

    def _set_raw_images(
        self, image: ArrayLike, thumbnail_source: ArrayLike
    ) -> None:
        """Set the image and its thumbnail.

        If floating point / grayscale then clip to [0..1].

        Parameters
        ----------
        image : ArrayLike
            Set this as the main image.
        thumbnail : ArrayLike
            Derive the thumbnail from this image.
        """
        super()._set_raw_images(image, thumbnail_source)

        # TODO_OCTREE: Create an octree as a test... the expection is this
        # is a *single* scale image and we create an octree on the fly just
        # so we have something to render.
        # with block_timer("create octree", print_time=True):
        self._octree = Octree.from_image(image)

        # None means use coarsest level.
        if self._octree_level is None:
            self._octree_level = self._octree.num_levels - 1

        # self._octree.print_tiles()

    @property
    def view_chunks(self):
        """Return the chunks currently in view."""

        # This will be None if we have not been drawn yet.
        if self._data_corners is None:
            return []

        # TODO_OCTREE: soon we will compute which level to draw.
        level = self._octree.levels[self._octree_level]

        return level.get_chunks(self._data_corners)
