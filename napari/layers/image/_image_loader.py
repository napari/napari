"""ImageLoader class.
"""
from ._image_slice_data import ImageSliceData


class ImageLoader:
    """The default synchronous ImageLoader."""

    def load(self, data: ImageSliceData) -> bool:
        """Load the ImageSliceData synchronously.

        Parameters
        ----------
        data : ImageSliceData
            The data to load.

        Returns
        -------
        bool
            True if load happened synchronously.
        """
        data.load_sync()
        return True

    def match(self, data: ImageSliceData) -> bool:
        """Return True if data matches what we are loading.

        Parameters
        ----------
        data : ImageSliceData
            Does this data match what we are loading?

        Returns
        -------
        bool
            Return True if data matches.
        """
        return True  # Always true for synchronous loader.
