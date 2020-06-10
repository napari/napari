from typing import NamedTuple, Optional, Tuple

import numpy as np

from ._image_view import ImageView
from ...types import ArrayLike, ImageConverter
from ...utils.chunk_loader import CHUNK_LOADER


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
        assert self.properties.multiscale

        return self.indices == indices

    def async_load(self, array: ArrayLike, indices=Tuple[slice, ...]):

        # Async for multiscale is not implemented yet.
        assert self.properties.multiscale

        if self.future is not None:
            # We switched slices so cancel the previous async load. This will
            # only cancel it if it was queue and the load was not in progress.
            self.future.cancel()

        # We are now loading a slice with these indices
        self.indices = indices
        self.future = CHUNK_LOADER.load_chunk(array)
        self.finished = False

    def has_loaded(self, array: ArrayLike) -> bool:
        """Check on image loading and install new image when available.
        """

        # Async for multiscale is not implemented yet.
        assert self.properties.multiscale

        # If no load is in progress: nothing to do.
        if self.future is None:
            return False

        # If still queued or loading, nothing to do.
        if not self.future.done():
            return False

        # Load has finished, but it could be either done or cancelled.
        self.finished = True

        if self.future.cancelled():
            # Not clear what to do here yet.
            self.future = None
            return False

        # Load succeeded, get the image that was loaded.
        image = self.future.result()
        image = image.transpose(self.properties.displayed_order)

        # They are the same for non-multiscale.
        thumbnail = image

        # This is now our slice contents.
        self.set_raw_image(image, thumbnail)
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
