from ...types import ArrayLike


class ImageView:
    """A raw image and a viewable version of it.

    A very simple class that groups together two related images, the raw one and
    the viewable one. Its primary purpose right now is just making sure the
    viewable image is updated when the raw one is changed. And just to group
    them together and provide convenient access.

    Attributes
    ----------
    _raw : ArrayLike
        The raw image.

    _view : ArrayLike
        The viewable image, dervied from raw.

    Example:
    --------
        # Set the viewable image and raw image to the same thing.
        image = ImageView(view_image)
        #  or
        image.view = view_image

        # Set the raw image, it will compute the new viewable one.
        image.raw = raw_image
    """

    def __init__(self, view_image: ArrayLike):
        """Create an ImageView with some default image.

        Parameters
        ----------
        view_image : ArrayLike
            Default viewable image, raw is set to the same thing.
        """
        self.view = view_image

    @property
    def view(self):
        """The viewable image."""
        return self._view

    @view.setter
    def view(self, view_image: ArrayLike):
        """Set the viewed and draw images.

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
        self._view = _raw_to_displayed(raw_image)


def _raw_to_displayed(raw_image: ArrayLike):
    """Determine displayed image from raw image.

    This is a NOOP right now for normal images.

    Parameters
    -------
    raw_image : ArrayLike
        Raw image.

    Returns
    -------
    view_image : ArrayLike
        Viewable image.
    """
    view_image = raw_image
    return view_image
