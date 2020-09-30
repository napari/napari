"""AsyncImageLoader classes.
"""
import logging
from typing import Optional

from ....components.experimental.chunk import ChunkKey
from .._image_loader import ImageLoader
from ._chunked_slice_data import ChunkedSliceData

LOGGER = logging.getLogger("napari.async")


class ChunkedImageLoader(ImageLoader):
    """Load images using the Chunkloader: synchronously or asynchronously.

    Attributes
    ----------
    current_key : Optional[ChunkKey]
        The ChunkKey we are currently loading or showing.
    """

    def __init__(self):
        # We're showing nothing to start.
        self.current_key: Optional[ChunkKey] = None

    def load(self, data: ChunkedSliceData):
        """Load this ChunkedSliceData (sync or async).

        Parameters
        ----------
        data : SlideData
            The data to load
        """
        key = ChunkKey(data.layer, data.indices)
        LOGGER.debug("AsyncImageLoader.load: %s", key)

        if self.current_key is not None and self.current_key == key:
            # We are already showing this slice, or its being loaded
            # asynchronously. TODO_ASYNC: does this happen?
            return None

        # Now "showing" this slice, even if it hasn't loaded yet.
        self.current_key = key

        if data.load_chunks(key):
            return data  # Load was sync.

        return None  # Load was async.

    def match(self, data: ChunkedSliceData) -> bool:
        """Return True if slice data matches what we are loading.

        Parameters
        ----------
        data : ChunkedSliceData
            Does this data match what we are loading?

        Return
        ------
        bool
            Return True if data matches.
        """
        key = data.request.key

        if self.current_key == key:
            LOGGER.debug("AsyncImageLoader.match: accept %s", key)
            return True

        # Probably we are scrolling through slices and we are no longer
        # showing this slice, so drop it. Even if we don't use it, it
        # should get into the cache, so the load wasn't totally wasted.
        LOGGER.debug("AsyncImageLoader.match: reject %s", key)
        return False
