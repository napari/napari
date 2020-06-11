from typing import NamedTuple, Tuple

import numpy as np

from ._image_view import ImageView
from ...types import ArrayLike, ImageConverter


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
    image_converter : ImageConverter
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
        image_converter: ImageConverter,
    ):
        self.image: ImageView = ImageView(view_image, image_converter)
        self.thumbnail: ImageView = ImageView(view_image, image_converter)
        self.properties = properties

        # We're showing the slice at these indices
        self.indices = None

    def contains(self, indices):
        return self.indices == indices

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

    def chunk_loaded(self, request):
        # Async not supported for multiscale yet
        assert not self.properties.multiscale

        image = request.array.transpose(self.properties.displayed_order)

        # Thumbnail is just the same image for non-multiscale.
        thumbnail = image

        # We are now showing data at these indices.
        self.set_raw_images(image, thumbnail)
        self.indices = request.indices
