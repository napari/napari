"""ImageSlice class.
"""
import logging
import os
from typing import Callable, Optional

import numpy as np

from ...types import ArrayLike
from ._image_slice_data import ImageSliceData
from ._image_view import ImageView
from ._sync_image_loader import SyncImageLoader
from .experimental._async_image_loader import AsyncImageLoader

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

        use_async = os.getenv("NAPARI_ASYNC", "0")
        self.loader = AsyncImageLoader() if use_async else SyncImageLoader()

        # With async there can be a gap between when the ImageSlice is
        # created and the data is actually loaded. However initialize
        # as True in case we aren't even doing async loading.
        self.loaded = True

    def _set_raw_images(
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
        # Single scale images don't have a separate thumbnail so we just
        # use the image itself.
        if thumbnail_source is None:
            thumbnail_source = image

        if self.rgb and image.dtype.kind == 'f':
            image = np.clip(image, 0, 1)
            thumbnail_source = np.clip(thumbnail_source, 0, 1)
        self.image.raw = image
        self.thumbnail.raw = thumbnail_source
        self.loaded = True

    def load(self, data: ImageSliceData) -> Optional[ImageSliceData]:
        """Load this data into the slice.

        Parameters
        ----------
        data : ImageSliceData
            The data to load into this slice.

        Return
        ------
        Optional[ImageSliceData]
            Return ImageSliceData if it was loaded synchronously.
        """
        self.loaded = False
        return self.loader.load(data)

    def on_loaded(self, data: ImageSliceData) -> bool:
        """Data was loaded, show the new data.

        Parameters
        ----------
        data : ImageSliceData
            The newly loaded data we want to show.

        Return
        ------
        bool
            True if the data was used, False if was for the wrong slice.
        """
        if not self.loader.match(data):
            return False

        # Display the newly loaded data.
        self._set_raw_images(data.image, data.thumbnail_source)
        return True
