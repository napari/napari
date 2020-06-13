from typing import NamedTuple, Tuple

import numpy as np

from ._image_view import ImageView
from ._image_text import get_text_image
from ...types import ArrayLike, ImageConverter
from ...utils.chunk_loader import ChunkRequest, CHUNK_LOADER


class ImageProperties(NamedTuple):
    multiscale: bool
    rgb: bool
    displayed_order: Tuple[int, ...]


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
        self.image: ImageView = ImageView(view_image, image_converter)
        self.thumbnail: ImageView = ImageView(view_image, image_converter)
        self.properties = properties

        # We're showing the slice at these indices
        self.current_indices = None

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
        print("ImageSlice.set_raw_images")

        if self.properties.rgb and image.dtype.kind == 'f':
            image = np.clip(image, 0, 1)
            thumbnail = np.clip(thumbnail, 0, 1)
        self.image.raw = image
        self.thumbnail.raw = thumbnail

    def load_chunk(self, request: ChunkRequest):
        """Load the requested chunk.

        Parameters
        ----------
        request : ChunkRequest
            This chunk was successfully loaded.
        """
        # Async not supported for multiscale yet
        assert not self.properties.multiscale

        if self.current_indices == request.indices:
            # We are loading or have loaded this slice already
            return

        # Save these so we don't try to re-load the same chunk.
        self.current_indices = request.indices

        # For now clear everything, later we'll only want to clear our layer?
        CHUNK_LOADER.clear_queued()

        # Load from cache or initiate async load.
        satisfied_request = CHUNK_LOADER.load_chunk(request)

        if satisfied_request is None:
            # Cache miss: async load was started. Show placeholder image
            # until the load finishes and self.chunk_loaded() is called.
            self._set_placeholder_image()
        else:
            # It was in the cache, put it to immediate use.
            self.chunk_loaded(satisfied_request)

    def chunk_loaded(self, request: ChunkRequest):
        """Chunk was loaded, show this new data.

        Parameters
        ----------
        request : ChunkRequest
            This chunk was successfully loaded.
        """
        print(f"ImageSlice.chunk_loaded {request.indices}")
        # Async not supported for multiscale yet
        assert not self.properties.multiscale

        # Is this the chunk we requested?
        if self.current_indices != request.indices:
            print(f"IGNORE CHUNK: {request.indices}")
            return

        # Could worker do the transpose? Does it take any time?
        image = request.array.transpose(self.properties.displayed_order)

        # Thumbnail is just the same image for non-multiscale.
        thumbnail = image

        # Show the new data, show this slice.
        self.set_raw_images(image, thumbnail)

    def _get_slice_string(self):
        """Get some string to describe the current slice.

        Right now it's just an integer like "5". This will be displayed on
        the placeholder image.
        """
        first = self.properties.displayed_order[0]
        return str(self.current_indices[first])

    def _set_placeholder_image(self):
        """Show placeholder until async load is finished.

        Today we only support async for non-multi-scale images. When the
        user navigates to a slice and we have no data to show, we show a
        "placeholder image" so we can stop drawing the previous slice and
        indicate that a load is in progress.

        Today our placeholder image is a just black with white text that
        says "loading: N" where N is the index of the slice being loaded.
        Long term we could show something more stylish or informative
        including some type of loading/progress animation.

        For multi-scale our "placeholder image" will often be imagery from
        a different level of detail. For example we can show coarser/blurry
        imagery until we can load the higher resolution data.

        This black screen with text is just for the worst case where we
        have flat nothing to show, but we don't want to keep drawing the
        previous slice.
        """
        text = f"loading: {self._get_slice_string()}"
        placeholder = get_text_image(text, self.properties.rgb)
        self.set_raw_images(placeholder, placeholder)
