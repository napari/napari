from typing import NamedTuple, Optional, Tuple

import numpy as np

from ._image_view import ImageView
from ...types import ArrayLike, ImageConverter
from ...utils.chunk_loader import CHUNK_LOADER, ChunkRequest


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
    image_converter : ImageConverter, optional
        ImageView uses this to convert from raw to viewable.
    rgb : bool
        True if the image RGB, as opposed to float/grayscale.
    displayed_order : Tuple[int, ...]
        The order of the displayed dimensions.

    Attributes
    ----------
    image : ImageView
        The main image for this slice.

    thumbnail : ImageView
        The thumbnail image for this slice.

    Example
    -------
        # Create with some default image.
        image_slice = ImageSlice(default_image)

        # Set raw image or thumbnail, viewable is computed.
        image_slice.image = raw_image
        image_slice.thumbnail = raw_thumbnail

        # Access the viewable images.
        draw_image(image_slice.image.view)
        draw_thumbnail(image_slice.thumbnail.view)
    """

    def __init__(
        self,
        view_image: ArrayLike,
        properties: ImageProperties,
        image_converter: Optional[ImageConverter] = None,
    ):
        self.image: ImageView = ImageView(view_image, image_converter)
        self.thumbnail: ImageView = ImageView(view_image, image_converter)
        self.properties = properties

        # No load is in progress to start.
        self.indices = None
        self.future = None
        self.finished = False

    def contains(self, indices) -> bool:
        # Async for multiscale is not implemented yet.
        assert not self.properties.multiscale

        return self.indices == indices

    def load_async(
        self, indices: Tuple[slice, ...], array: ArrayLike, callback
    ) -> None:

        # Async for multiscale is not implemented yet.
        assert not self.properties.multiscale

        if self.future is not None:
            # We switched slices so cancel the previous async load. This will
            # only cancel it if it was queue and the load was not in progress.
            self.future.cancel()

        # We are now loading a slice with these indices
        self.indices = indices
        request = ChunkRequest(indices, array, callback)
        self.future = CHUNK_LOADER.load_chunk(request)
        self.finished = False

    def on_loaded(self):
        print("ImageSlice.on_loaded")

    def has_loaded(self, array: ArrayLike) -> bool:
        """Check on image loading and install new image when available.
        """
        print("has_loaded")
        # Async for multiscale is not implemented yet.
        assert not self.properties.multiscale

        # If no load is in progress: nothing to do.
        if self.future is None:
            print("has_loaded: no future")
            return False

        # If still queued or loading, nothing to do.
        if not self.future.done():
            print("has_loaded: future not done")
            return False

        # Load has finished, but it could be either done or cancelled.
        self.finished = True

        if self.future.cancelled():
            # Not clear what to do here yet.
            print("has_loaded: CANCELLED")
            self.future = None
            return False

        print("has_loaded: LOADED")

        # Load succeeded, get the image that was loaded.
        request = self.future.result()
        image = request.array.transpose(self.properties.displayed_order)

        # They are the same for non-multiscale.
        thumbnail = image

        # This is now our slice contents.
        self.set_raw_images(image, thumbnail)
        return True

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
