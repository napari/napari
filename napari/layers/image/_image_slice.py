from ...types import ArrayLike
from ._image_view import ImageView


class ImageSlice:
    """The slice of the image that we are currently viewing.

    Right now this just holds the image and its thumbnail, however future async
    and multiscale-async changes will likely grow this class a lot.

    Example
    -------
        # Create with some default image.
        slice = ImageSlice(default_image)

        # Set raw image or thumbnail, viewable is computed.
        slice.image = raw_image
        slice.thumbnail = raw_thumbnail

        # Access the viewable images.
        draw_image(slice.image.view)
        draw_thumbnail(slice.thumbnail.view)
    """

    def __init__(self, view_image: ArrayLike):
        """
        Create an ImageSlice with some default viewable image.

        Parameters
        ----------
        view_image : ArrayLike
            Our default viewable image and viewable thumbnail.
        """
        self.image = ImageView(view_image)
        self.thumbnail = ImageView(view_image)
