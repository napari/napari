"""ChunkedSliceData class.

This is for pre-Octree Image class only.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from napari.components.experimental.chunk import ChunkRequest, chunk_loader
from napari.layers.base import Layer
from napari.layers.image._image_slice_data import ImageSliceData
from napari.layers.image.experimental._image_location import ImageLocation

LOGGER = logging.getLogger("napari.loader")

if TYPE_CHECKING:
    from napari.types import ArrayLike


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
    ) -> None:
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

    def load_chunks(self) -> bool:
        """Load this slice data's chunks sync or async.

        Returns
        -------
        bool
            True if chunks were loaded synchronously.
        """
        # Always load the image.
        chunks = {'image': self.image}

        # Optionally load th e thumbnail_source if it exists.
        if self.thumbnail_source is not None:
            chunks['thumbnail_source'] = self.thumbnail_source

        def _should_cancel(chunk_request: ChunkRequest) -> bool:
            """Cancel any requests for this same data_id.

            The must be requests for other slices, but we only ever show
            one slice at a time, so they are stale.
            """
            return chunk_request.location.data_id == id(self.image)

        # Cancel loads for any other data_id/slice besides this one.
        chunk_loader.cancel_requests(_should_cancel)

        # Create the request and load it.
        location = ImageLocation(self.layer, self.indices)
        self.request = ChunkRequest(location, chunks)
        satisfied_request = chunk_loader.load_request(self.request)

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
        indices = request.location.indices
        image = request.chunks.get('image')
        thumbnail_slice = request.chunks.get('thumbnail_slice')
        return cls(layer, indices, image, thumbnail_slice, request)
