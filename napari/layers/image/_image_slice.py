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
        The initial image for the time and its thumbail.
    properties : Image_Properties
        We are displaying a sliced from an Image with the properties.
    image_converter : ImageConverter
        ImageView uses this to convert from raw to viewable.

    Attributes
    ----------
    image : ImageView
        The main image for this slice.

    thumbnail : ImageView
        The source image used to compute the smaller thumbnail image.

    Examples
    --------
    Create with some default image:

    >> image_slice = ImageSlice(default_image)

    Set raw image or thumbnail, viewable is computed.:

    >> image_slice.image = raw_image
    >> image_slice.thumbnail = raw_thumbnail

    Access viewable images:

    >> draw_image(image_slice.image.view)
    >> draw_thumbnail(image_slice.thumbnail.view)
    """

    def __init__(
        self,
        view_image: ArrayLike,
        properties: ImageProperties,
        image_converter: ImageConverter,
    ):
        LOGGER.info("ImageSlice.__init__")
        self.image: ImageView = ImageView(view_image, image_converter)
        self.thumbnail: ImageView = ImageView(view_image, image_converter)
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

        # If ChunkLoader is synchronous or the chunk is cached, this will
        # satisfy the request right away, otherwise it will initiate an
        # async load in a worker thread.
        satisfied_request = chunk_loader.load_chunk(request)

        if satisfied_request is not None:
            self.chunk_loaded(satisfied_request)

    def chunk_loaded(self, request: ChunkRequest) -> bool:
        """Chunk was loaded, show this new data.

        Parameters
        ----------
        request : ChunkRequest
            This chunk was successfully loaded.

        Return
        ------
        bool
            False if the chunk was for the wrong slice.
        """
        LOGGER.info("ImageSlice.chunk_loaded: %s", request.key)

        # Is this the chunk we requested?
        if not np.all(self.current_indices == request.indices):
            LOGGER.warn(
                "ImageSlice.chunk_loaded: IGNORE CHUNK %s", request.key
            )
            return False

        order = self.properties.displayed_order

        image = request.chunks['image'].transpose(order)

        try:
            thumbnail_source = request.chunks['thumbnail_source'].transpose(
                order
            )
        except KeyError:
            # No explicit thumbnail_source so use the image (single-scale?)
            thumbnail_source = image

        # Show the new data, show this slice.
        self.set_raw_images(image, thumbnail_source)
        return True
