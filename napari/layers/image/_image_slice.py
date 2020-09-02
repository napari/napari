"""ImageSlice class.
"""
import logging
from typing import Callable, Optional

import numpy as np

from ...components.chunk import ChunkKey, ChunkRequest, chunk_loader
from ...types import ArrayLike
from ._image_view import ImageView

LOGGER = logging.getLogger("napari.async")


class ImageSlice:
    """The slice of the image that we are currently viewing.

    Parameters
    ----------
    image : ArrayLike
        The initial image used as the image and the thumbnail source.
    image_converter : Callable[[ArrayLike], ArrayLike]
        ImageView uses this to convert from raw to viewable.
    rgb : bool
        True if the image is RGB format. Otherwise its RGBA.

    Attributes
    ----------
    image : ImageView
        The main image for this slice.
    thumbnail : ImageView
        The source image used to compute the smaller thumbnail image.
    rgb : bool
        Is the image in RGB or RGBA format.
    current_key : Optional[ChunkKey]
        The ChunkKey we are currently showing or which is loading.
    loaded : bool
        Has the data for this slice been loaded yet.
    """

    def __init__(
        self,
        image: ArrayLike,
        image_converter: Callable[[ArrayLike], ArrayLike],
        rgb: bool = False,
    ):
        LOGGER.debug("ImageSlice.__init__")
        self.image: ImageView = ImageView(image, image_converter)
        self.thumbnail: ImageView = ImageView(image, image_converter)
        self.rgb = rgb

        # We're showing nothing to start.
        self.current_key: Optional[ChunkKey] = None

        # With async there can be a gap between when the ImageSlice is
        # created and the data is actually loaded.
        self.loaded = False

    def set_raw_images(
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
        if self.rgb and image.dtype.kind == 'f':
            image = np.clip(image, 0, 1)
            thumbnail_source = np.clip(thumbnail_source, 0, 1)
        self.image.raw = image
        self.thumbnail.raw = thumbnail_source
        self.loaded = True

    def load_chunk(self, request: ChunkRequest) -> Optional[ChunkRequest]:
        """Load the requested chunk asynchronously.

        Parameters
        ----------
        request : ChunkRequest
            Load this chunk sync or async.
        """
        LOGGER.debug("ImageSlice.load_chunk: %s", request.key)

        # Now "showing" this slice, even if it hasn't loaded yet.
        self.current_key = request.key
        self.loaded = False

        # This will return a satisfied request if ChunkLoader is doing
        # synchronous loading or the chunk was in the cache. If it returns
        # None that means a request was queued and it will be loaded in a
        # worker thread or process.
        return chunk_loader.load_chunk(request)

    def on_chunk_loaded(self, request: ChunkRequest) -> bool:
        """Chunk was loaded, show this new data.

        Parameters
        ----------
        request : ChunkRequest
            The chunk request that was loaded in a worker thread or process.

        Return
        ------
        bool
            False if the chunk was for the wrong slice and was not used.
        """
        # Is this the chunk we last requested?
        if not self.current_key == request.key:
            # Probably we are scrolling through slices and we are no longer
            # showing this slice, so drop it. It should have been added
            # to the cache so the load was not totally wasted.
            LOGGER.debug("ImageSlice.chunk_loaded: reject %s", request.key)
            return False

        LOGGER.debug("ImageSlice.chunk_loaded: accept %s", request.key)

        # Display the newly loaded data.
        self.set_raw_images(request.image, request.thumbnail_source)
        return True
