from ...types import ArrayLike
from ._image_view import ImageView


class ImageSlice:
    """The slice of the image that we are currently viewing.

    Right now this just holds the image and its thumbnail, however future async
    and multiscale-async changes will likely grow this class a lot.

    Attributes
    ----------
    image : ImageView
        The main image for this slice.

    thumbnail : ImageView
        The smaller thumbnail image for this slice.

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
        self, view_image: ArrayLike, converter: ImageView.ConverterType = None
    ):
        """
        Create an ImageSlice with some default viewable image.

        Parameters
        ----------
        view_image : ArrayLike
            The default image for the time and its thumbail.
        converter : ImageView.ConverterType
            ImageView will use this to convert from raw to viewable.
        """
        self.image: ImageView = ImageView(view_image, converter)
        self.thumbnail: ImageView = ImageView(view_image, converter)
