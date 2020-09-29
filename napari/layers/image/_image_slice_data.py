"""ImageSliceData class.
"""
import logging

import numpy as np

from ...components.chunk import ChunkKey, ChunkRequest, chunk_loader
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

    def load(self) -> None:
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
            self.thumbanail_source = self.thumbanail_source.transpose(order)

    def load_chunks(self, key: ChunkKey) -> bool:
        """Load chunks sync or async.

        Return
        ------
        bool
            True if chunks were loaded synchronously.
        """
        # Always load the image.
        chunks = {'image': self.image}

        # Optionally also load the thumbnail_source.
        if self.thumbnail_source is not None:
            chunks['thumbnail_source'] = self.thumbnail_source

        # Create our ChunkRequest.
        self.chunk_request = chunk_loader.create_request(
            self.layer, key, chunks
        )

        # Load using the global ChunkLoader.
        satisfied_request = chunk_loader.load_chunk(self.chunk_request)

        if satisfied_request is None:
            return False  # load was async

        self.chunk_request = satisfied_request
        return True  # load was sync

    @property
    def rgb(self) -> bool:
        return guess_rgb(self.image.shape)

    @classmethod
    def from_request(cls, layer, request: ChunkRequest):
        image = request.chunks.get('image')
        thumbnail_slice = request.chunks.get('thumbnail_slice')
        indices = request.key.indices
        return cls(layer, indices, image, thumbnail_slice, request)
