"""ImageSliceData class.
"""
import logging

import numpy as np

from ._image_utils import guess_rgb

LOGGER = logging.getLogger("napari.async")


class ImageSliceData:
    def __init__(
        self, layer, image_indices, image, thumbnail_source, chunk_request=None
    ):
        self.layer = layer
        self.indices = image_indices
        self.image = image
        self.thumbnail_source = thumbnail_source

        # chunk_request is None unless doing asynchronous loading.
        self.chunk_request = chunk_request

    def load_sync(self) -> None:
        """Call asarray on our images to load them."""
        self.image = np.asarray(self.image)

        if self.thumbnail_source is not None:
            self.thumbnail_source = np.asarray(self.thumbnail_source)

    def transpose(self, order: tuple) -> None:
        """Transpose our images.

        Parameters
        ----------
        order : tuple
            Transpose the image into this order.
        """
        self.image = self.image.transpose(order)

        if self.thumbnail_source is not None:
            self.thumbnail_source = self.thumbnail_source.transpose(order)

    @property
    def rgb(self) -> bool:
        return guess_rgb(self.image.shape)
