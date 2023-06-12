"""ChunkedImageLoader class.

This is for pre-Octree Image class only.
"""
import logging
from typing import Optional

from napari.layers.image._image_loader import ImageLoader
from napari.layers.image.experimental._chunked_slice_data import (
    ChunkedSliceData,
)
from napari.layers.image.experimental._image_location import ImageLocation

LOGGER = logging.getLogger("napari.loader")


class ChunkedImageLoader(ImageLoader):
    """Load images using the Chunkloader: synchronously or asynchronously.

    Attributes
    ----------
    _current : Optional[ImageLocation]
        The location we are currently loading or showing.
    """

    def __init__(self) -> None:
        # We're showing nothing to start.
        self._current: Optional[ImageLocation] = None

    def load(self, data: ChunkedSliceData) -> bool:
        """Load this ChunkedSliceData (sync or async).

        Parameters
        ----------
        data : ChunkedSliceData
            The data to load

        Returns
        -------
        bool
            True if load happened synchronously.
        """
        location = ImageLocation(data.layer, data.indices)

        LOGGER.debug("ChunkedImageLoader.load")

        if self._current is not None and self._current == location:
            # We are already showing this slice, or its being loaded
            # asynchronously.
            return False

        # Now "showing" this slice, even if it hasn't loaded yet.
        self._current = location

        if data.load_chunks():
            return True  # Load was sync, load is done.

        return False  # Load was async, so not loaded yet.

    def match(self, data: ChunkedSliceData) -> bool:
        """Return True if slice data matches what we are loading.

        Parameters
        ----------
        data : ChunkedSliceData
            Does this data match what we are loading?

        Returns
        -------
        bool
            Return True if data matches.
        """
        location = data.request.location

        if self._current == location:
            LOGGER.debug("ChunkedImageLoader.match: accept %s", location)
            return True

        # Data was for a slice we are no longer looking at.
        LOGGER.debug("ChunkedImageLoader.match: reject %s", location)
        return False
