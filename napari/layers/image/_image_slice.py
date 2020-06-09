from typing import NamedTuple, Optional, Tuple
from enum import Enum

import numpy as np

from ._image_view import ImageView
from ...types import ArrayLike, ImageConverter
from ...utils.chunk_loader import CHUNK_LOADER


class ImageProperties(NamedTuple):
    multiscale: bool
    rgb: bool
    displayed_order: Tuple[int, ...]


class SliceStatus(Enum):
    EMPTY = 1
    WAITING = 2
    LOADED = 3
    CANCELLED = 3


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

        # We start out empty (except for our default image),
        self.status = SliceStatus.EMPTY
        self.future = None

    def empty(self) -> bool:
        """Return True if this slice is currently empty."""
        return self.status == SliceStatus.EMPTY

    def waiting(self) -> bool:
        """Return True if this slice is waiting on a chunk to load."""
        return self.status == SliceStatus.WAITING

    def loaded(self) -> bool:
        """Return True if the slice is fully loaded."""
        return self.status == SliceStatus.LOADED

    def async_load(self, array: ArrayLike):
        self.future = CHUNK_LOADER.load_chunk(array)
        self.status = SliceStatus.LOADING

    def update(self) -> None:

        # Async is only for non-multi-scale so far.
        assert self.properties.multiscale

        # Only proceed if we were waiting and finish.
        if not self.waiting() or not self.future.done():
            return

        # If it's done because it was cancelled.
        if self.future.cancelled():
            self.status = SliceStatus.CANCELLED
            return

        # It's done and succeeded, get the image that was loaded.
        image = self.future.result()
        image = image.transpose(self.properties.displayed_order)

        # They are the same for non-multiscale.
        thumbnail = image

        self.set_images_with_clipping(image, thumbnail)

    def set_raw_images(self, image: ArrayLike, thumbnail: ArrayLike) -> None:
        """Set the image and its thumbnail.

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
