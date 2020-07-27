import logging
from typing import NamedTuple, Tuple

import numpy as np

from ...types import ArrayLike, ImageConverter
from ...utils.chunk import ChunkRequest, chunk_loader

from ._image_view import ImageView

LOGGER = logging.getLogger("ChunkLoader")


class ImageProperties(NamedTuple):
    multiscale: bool
    rgb: bool
    ndim: int
    displayed_order: Tuple[slice, ...]


class ImageSlice:
    """The slice of the image that we are currently viewing.

    Right now this just holds the image and its thumbnail, however future async
    and multiscale-async changes will likely grow this class a lot.

    Parameters
    ----------
    view_image : ArrayLike
        The initial image used as the image and the thumbnail source.
    properties : Image_Properties
        The Image we are slicing has these properties.
    image_converter : ImageConverter
        ImageView uses this to convert from raw to viewable.

    Attributes
    ----------
    image : ImageView
        The main image for this slice.

    thumbnail : ImageView
        The source image used to compute the smaller thumbnail image.
    """

    def __init__(
        self,
        view_image: ArrayLike,
        image_converter: ImageConverter,
        properties: ImageProperties = None,
    ):
        LOGGER.info("ImageSlice.__init__")
        self.image: ImageView = ImageView(view_image, image_converter)
        self.thumbnail: ImageView = ImageView(view_image, image_converter)

        # If None then we are in legacy mode, for the old Image.py, and
        # it cannot call our set_raw_images() or chunk_loaded() methods
        # which use these properties.
        self.properties = properties

        # We're showing the slice at these indices.
        self.current_indices = None

        # With async there can be a gap between when the ImageSlice is
        # created and the data is actually loaded.
        self.loaded = False

    def set_raw_images(self, image: ArrayLike, thumbnail: ArrayLike) -> None:
        """Set the image and its thumbnail.

        If floating point / grayscale then clip to [0..1].

        Parameters
        ----------
        image : ArrayLike
            Set this as the main image.
        thumbnail : ArrayLike
            Set this as the thumbnail.
        """
        if self.properties.rgb and image.dtype.kind == 'f':
            image = np.clip(image, 0, 1)
            thumbnail = np.clip(thumbnail, 0, 1)
        self.image.raw = image
        self.thumbnail.raw = thumbnail
        self.loaded = True

    def load_chunk(self, request: ChunkRequest) -> None:
        """Load the requested chunk asynchronously.

        Parameters
        ----------
        request : ChunkRequest
            Load this chunk sync or async.
        """
        LOGGER.info("ImageSlice.load_chunk: %s", request.key)

        # Now "showing" this slice, even if it hasn't loaded yet.
        self.current_indices = request.indices

        # This will return a satisfied request in ChunkLoader is doing
        # syncrhonous loading or the chunk was in the cache. If it returns
        # None that means a request was queued and it will be loaded in a
        # worker thread or process.
        return chunk_loader.load_chunk(request)

    def chunk_loaded(self, request: ChunkRequest) -> bool:
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
        LOGGER.info("ImageSlice.chunk_loaded: %s", request.key)

        # Is this the chunk we requested?
        if not np.all(self.current_indices == request.indices):
            LOGGER.warn(
                "ImageSlice.chunk_loaded: IGNORE CHUNK %s", request.key
            )
            return False

        order = self.properties.displayed_order
        chunks = request.chunks
        image = chunks['image'].transpose(order)

        try:
            thumbnail_source = chunks['thumbnail_source'].transpose(order)
        except KeyError:
            # We use the image as the thumbnail_source for single-scale.
            thumbnail_source = image

        # Display the newly loaded data.
        self.set_raw_images(image, thumbnail_source)
        return True
