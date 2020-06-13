from ...types import ArrayLike, ImageConverter


class ImageView:
    """A raw image and a viewable version of it.

    A very simple class that groups together two related images, the raw one and
    the viewable one. Its primary purpose right now is just making sure the
    viewable image is updated when the raw one is changed. And just to group
    them together and provide convenient access.

    Parameters
    ----------
    view_image : ArrayLike
        Default viewable image, raw is set to the same thing.
    image_converter : Optional[ImageConverter]
        If given this is used to convert images from raw to viewable.

    Attributes
    ----------
    _raw : ArrayLike
        The raw image.

    _view : ArrayLike
        The viewable image, dervied from raw.

    Examples
    --------
    Create ImageView with initial default:

    >> image = ImageView(view_image)

    Update ImageView's raw image, it will compute the new viable one:

    >> image.raw = raw_image
    """

    def __init__(self, view_image: ArrayLike, image_converter: ImageConverter):
        """Create an ImageView with some default image.
        """
        self.view = view_image
        self.image_converter = image_converter

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

        # Update the view image based on this new raw image.
        self._view = self.image_converter(raw_image)
