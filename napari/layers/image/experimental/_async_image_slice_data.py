"""ImageSliceData class.
"""
import logging

from ....components.experimental.chunk import (
    ChunkKey,
    ChunkRequest,
    chunk_loader,
)
from .._image_slice_data import ImageSliceData

LOGGER = logging.getLogger("napari.async")


class AsyncImageSliceData(ImageSliceData):
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

    @classmethod
    def from_request(cls, layer, request: ChunkRequest):
        image = request.chunks.get('image')
        thumbnail_slice = request.chunks.get('thumbnail_slice')
        indices = request.key.indices
        return cls(layer, indices, image, thumbnail_slice, request)
