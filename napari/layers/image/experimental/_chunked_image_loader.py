"""ChunkedImageLoader classes.
"""
import logging
from typing import Optional

from ....components.experimental.chunk import ChunkKey, LayerKey
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

    def load(self, data: ChunkedSliceData) -> bool:
        """Load this ChunkedSliceData (sync or async).

        Parameters
        ----------
        data : ChunkedSliceData
            The data to load

        Return
        ------
        bool
            True if load happened synchronously.
        """
        layer = data.layer
        layer_key = LayerKey.from_layer(layer, data.indices)
        key = ChunkKey(layer_key)

        LOGGER.debug("ChunkedImageLoader.load: %s", key)

        if self.current_key is not None and self.current_key == key:
            # We are already showing this slice, or its being loaded
            # asynchronously. TODO_ASYNC: does this still happen?
            return False

        # Now "showing" this slice, even if it hasn't loaded yet.
        self.current_key = key

        if data.load_chunks(key):
            return True  # Load was sync, load is done.

        return False  # Load was async, so not loaded yet.

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
            LOGGER.debug("ChunkedImageLoader.match: accept %s", key)
            return True

        # Probably we are scrolling through slices and we are no longer
        # showing this slice, so drop it. Even if we don't use it, it
        # should get into the cache, so the load wasn't totally wasted.
        LOGGER.debug("ChunkedImageLoader.match: reject %s", key)
        return False
