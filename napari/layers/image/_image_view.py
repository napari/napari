"""ImageView class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from napari.types import ArrayLike, Callable


class ImageView:
    """A raw image and a viewable version of it.

    Small class that groups together two related images, the raw one and
    the viewable one. The image_converter passed in is either
    Image._raw_to_displayed or Labels._raw_to_displayed. The Image one does
    nothing but the Labels ones does colormapping.

    Parameters
    ----------
    view_image : ArrayLike
        Default viewable image, raw is set to the same thing.
    image_converter : Callable[[ArrayLike], ArrayLike]
        Used to convert images from raw to viewable.

    Attributes
    ----------
    view : ArrayLike
        The raw image.

    image_convert : ImageConvert
        Converts from raw to viewable.
    """

    def __init__(
        self,
        view_image: ArrayLike,
        image_converter: Callable[[ArrayLike], ArrayLike],
    ):
        """Create an ImageView with some default image."""
        self.view = view_image
        self.image_converter = image_converter

    @property
    def view(self):
        """The viewable image."""
        return self._view

    @view.setter
    def view(self, view_image: ArrayLike):
        """Set the viewed and raw image.

        Parameters
        ----------
        view_image : ArrayLike
            The viewable and raw images are set to this.
        """
        self._view = view_image
        self._raw = view_image

    @property
    def raw(self):
        """The raw image."""
        return self._raw

    @raw.setter
    def raw(self, raw_image: ArrayLike):
        """Set the raw image, viewable image is computed.

        Parameters
        ----------
        raw_image : ArrayLike
            The raw image to set.
        """
        self._raw = raw_image

        # Update the view image based on this new raw image.
        self._view = self.image_converter(raw_image)
