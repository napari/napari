from typing import Optional

from ._image_view import ImageView
from ...types import ArrayLike, ImageConverter


class ImageSlice:
    """The slice of the image that we are currently viewing.

    Right now this just holds the image and its thumbnail, however future async
    and multiscale-async changes will likely grow this class a lot.

    Parameters
    ----------
    view_image : ArrayLike
        The default image for the time and its thumbail.
    image_converter : ImageConverter, optional
        ImageView uses this to convert from raw to viewable.

    Attributes
    ----------
    image : ImageView
        The main image for this slice.

    thumbnail : ImageView
        The smaller thumbnail image for this slice.

    Examples
    --------
    Create with some default image:

    >> image_slice = ImageSlice(default_image)

    Set raw image or thumbnail, viewable is computed.:

    >> image_slice.image = raw_image
    >> image_slice.thumbnail = raw_thumbnail

    Access viewable images:

    >> draw_image(image_slice.image.view)
    >> draw_thumbnail(image_slice.thumbnail.view)
    """

    def __init__(
        self,
        view_image: ArrayLike,
        image_converter: Optional[ImageConverter] = None,
    ):
        """
        Create an ImageSlice with some default viewable image.
        """
        self.image: ImageView = ImageView(view_image, image_converter)
        self.thumbnail: ImageView = ImageView(view_image, image_converter)
