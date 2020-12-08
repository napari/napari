"""ChunkedSliceData class.
"""
import logging
from typing import Optional

from ....components.experimental.chunk import (
    ChunkKey,
    ChunkRequest,
    LayerRef,
    chunk_loader,
)
from ....types import ArrayLike
from ...base import Layer
from .._image_slice_data import ImageSliceData

LOGGER = logging.getLogger("napari.async")


class ChunkedSliceData(ImageSliceData):
    """SliceData that works with ChunkLoader.

    Parameters
    ----------
    layer : Layer
        The layer that contains the data.
    indices : Tuple[Optional[slice], ...]
        The indices of this slice.
    image : ArrayList
        The image to display in the slice.
    thumbnail_source : ArrayList
        The source used to create the thumbnail for the slice.
    request : Optional[ChunkRequest]
        The ChunkRequest that was used to load this data.
    """

    def __init__(
        self,
        layer: Layer,
        indices,
        image: ArrayLike,
        thumbnail_source: ArrayLike,
        request: Optional[ChunkRequest] = None,
    ):
        super().__init__(layer, indices, image, thumbnail_source)

        # When ChunkedSliceData is first created self.request is
        # None, it will get set one of two ways:
        #
        # 1. Synchronous load: our load_chunks() method will set
        #    self.request with the satisfied ChunkRequest.
        #
        # 2. Asynchronous load: Image.on_chunk_loaded() will create
        #    a new ChunkedSliceData using our from_request()
        #    classmethod. It will set the completed self.request.
        #
        self.request = request
        self.thumbnail_image = None

    def load_chunks(self, key: ChunkKey) -> bool:
        """Load this slice data's chunks sync or async.

        Parameters
        ----------
        key : ChunkKey
            The key for the chunks we are going to load.

        Return
        ------
        bool
            True if chunks were loaded synchronously.
        """
        # Always load the image.
        chunks = {'image': self.image}

        # Optionally load th e thumbnail_source if it exists.
        if self.thumbnail_source is not None:
            chunks['thumbnail_source'] = self.thumbnail_source

        # Create the ChunkRequest and load it with the ChunkLoader.
        layer_ref = LayerRef.create_from_layer(self.layer, self.indices)
        self.request = chunk_loader.create_request(layer_ref, key, chunks)
        satisfied_request = chunk_loader.load_chunk(self.request)

        if satisfied_request is None:
            return False  # Load was async.

        # Load was sync.
        self.request = satisfied_request
        self.image = self.request.chunks.get('image')
        self.thumbnail_image = self.request.chunks.get('thumbnail_source')
        return True

    @classmethod
    def from_request(cls, layer: Layer, request: ChunkRequest):
        """Create an ChunkedSliceData from a ChunkRequest.

        Parameters
        ----------
        layer : Layer
            The layer for this request.

        request : ChunkRequest
            The request that was loaded.
        """
        indices = request.key.layer_key.indices
        image = request.chunks.get('image')
        thumbnail_slice = request.chunks.get('thumbnail_slice')
        return cls(layer, indices, image, thumbnail_slice, request)
