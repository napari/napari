from typing import NamedTuple, Tuple

import numpy as np

from ._image_view import ImageView
from ...types import ArrayLike, ImageConverter
from ...utils.chunk_loader import ChunkRequest, CHUNK_LOADER


def get_text_image(text, rgb):
    """For debugging create an image with some text on it.
    """
    from PIL import Image, ImageDraw, ImageFont

    size = (1024, 1024)
    if rgb:
        image = Image.new('RGB', size)
    else:
        image = Image.new('L', size)

    text = str(text)

    font = ImageFont.truetype('Arial Black.ttf', size=72)
    (width, height) = font.getsize(text)
    x = (image.width / 2) - width / 2
    y = (image.height / 2) - height / 2

    color = 'rgb(255, 255, 255)'  # white color
    draw = ImageDraw.Draw(image)
    draw.text((x, y), text, fill=color, font=font)

    return np.array(image)


class ImageProperties(NamedTuple):
    multiscale: bool
    rgb: bool
    displayed_order: Tuple[int, ...]


class ImageSlice:
    """The slice of the image that we are currently viewing.

    Right now this just holds the image and its thumbnail, however future async
    and multiscale-async changes will likely grow this class a lot.

    Parameters
    ----------
    view_image : ArrayLike
        The initial image for the time and its thumbail.
    properties : Image_Properties
        We are displaying a sliced from an Image with the properties.
    image_converter : ImageConverter
        ImageView uses this to convert from raw to viewable.

    Attributes
    ----------
    image : ImageView
        The main image for this slice.

    thumbnail : ImageView
        The thumbnail image for this slice.

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
        self,
        view_image: ArrayLike,
        properties: ImageProperties,
        image_converter: ImageConverter,
    ):
        self.image: ImageView = ImageView(view_image, image_converter)
        self.thumbnail: ImageView = ImageView(view_image, image_converter)
        self.properties = properties

        # We're showing the slice at these indices
        self.current_indices = None

    def set_raw_images(self, image: ArrayLike, thumbnail: ArrayLike) -> None:
        """Set the image and its thumbnail.

        If floating point / grayscale then clip to [0..1].

        Parameters
        ----------
        image : ArrayLike
            Set this as the main image.
        thumbnail : ArrayLike
            Set this as the thumbnail.
        """
        print("ImageSlice.set_raw_images")

        if self.properties.rgb and image.dtype.kind == 'f':
            image = np.clip(image, 0, 1)
            thumbnail = np.clip(thumbnail, 0, 1)
        self.image.raw = image
        self.thumbnail.raw = thumbnail

    def load_chunk(self, request: ChunkRequest):
        """Load the requested chunk.

        Parameters
        ----------
        request : ChunkRequest
            This chunk was successfully loaded.
        """
        # Async not supported for multiscale yet
        assert not self.properties.multiscale

        if self.current_indices == request.indices:
            # We are loading or have loaded this slice already
            return

        # Initiate the async load, self.chunk_loaded() will be called when ready.
        CHUNK_LOADER.load_chunk(request)

        # Save these so we don't try to re-load the same chunk.
        self.current_indices = request.indices

        # While still debugging set a blank numbered image so something
        # happens and we know an image is being loaded.
        self._set_index_image()

    def chunk_loaded(self, request: ChunkRequest):
        """Chunk was loaded, show this new data.

        Parameters
        ----------
        request : ChunkRequest
            This chunk was successfully loaded.
        """
        print(f"ImageSlice.chunk_loaded {request.indices}")
        # Async not supported for multiscale yet
        assert not self.properties.multiscale

        # Is this the chunk we requested?
        if self.current_indices != request.indices:
            print(f"IGNORE CHUNK: {request.indices}")
            return

        # Could worker do the transpose? Does it take any time?
        image = request.array.transpose(self.properties.displayed_order)

        # Thumbnail is just the same image for non-multiscale.
        thumbnail = image

        # Show the new data, show this slice.
        self.set_raw_images(image, thumbnail)

    def _get_current_index(self):
        """Get slice index for debug placeholder image."""
        # Not positive this is right in all cases...
        first = self.properties.displayed_order[0]
        return self.current_indices[first]

    def _set_index_image(self):
        """For debugging set a slice image that has text on it.
        """
        print(f"{self.properties.displayed_order}")
        index = self._get_current_index()
        image = get_text_image(f"loading: {index}", self.properties.rgb)
        self.set_raw_images(image, image)
