"""ImageLoader class.
"""
from ._image_slice_data import ImageSliceData


class ImageLoader:
    """The default synchrnous ImageLoader."""

    def load(self, data: ImageSliceData) -> ImageSliceData:
        data.load_sync()
        return data

    def match(self, data: ImageSliceData) -> bool:
        """Return True if data matches what we are loading.

        Parameters
        ----------
        data : ImageSliceData
            Does this data match what we are loading?

        Return
        ------
        bool
            Return True if data matches.
        """
        return True  # Always true for synchronous loader.
