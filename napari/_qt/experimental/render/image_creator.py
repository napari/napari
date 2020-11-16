"""Create test images

This is a throw-away file for creating a test image for octree rendering
development. If we keep test images in the product long term we'll
have a nicer way to generate them.

Long term we probably do not want to use PIL for example.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ....layers.image.experimental import (
    ImageConfig,
    create_multi_scale_from_image,
)
from ....utils.perf import block_timer


def draw_text_grid(image, text: str) -> None:
    """Draw some text into the given image in a grid.

    Parameters
    ----------
    test : str
        The text to draw. For example a slice index like "3".
    """

    try:
        font = ImageFont.truetype('Arial Black.ttf', size=74)
    except OSError:
        font = ImageFont.load_default()
    (text_width, text_height) = font.getsize(text)

    color = 'rgb(255, 255, 255)'  # white
    draw = ImageDraw.Draw(image)

    width, height = image.size

    text_size = 100  # approx guess with some padding

    rows, cols = int(height / text_size), int(width / text_size)

    for row in range(rows + 1):
        for col in range(cols + 1):
            y = (row / rows) * image.height - text_height / 2
            x = (col / cols) * image.width - text_width / 2

            draw.text((x, y), text, fill=color, font=font)
    draw.rectangle([0, 0, image.width, image.height], outline=color, width=5)


def create_test_image(text, config: ImageConfig) -> np.ndarray:
    """Create a test image for testing tiled rendering.

    The test image just has digits all over it. The digits will typically
    be used to show the slice number like "0" or "42".

    image_shape: Tuple[int, int]
        The [height, width] shape of the image.
    """
    text = str(text)  # Might be an int.

    # Image.new wants (width, height) so swap them.
    image_size = config.image_shape[::-1]

    # Create the image, draw on the text, return it.
    image = Image.new('RGB', image_size)
    draw_text_grid(image, text)
    return np.array(image)


def create_test_image_multi(text, image_config: ImageConfig) -> np.ndarray:
    """Create a multiscale test image for testing tiled rendering.

    The test image is blank with digits all over it. The digits will
    typically be used to show the slice number. Should do something fancier
    with colors and less repetition.

    image_config: ImageConfig
        The shape and other details about the image to be created.
    """
    text = str(text)  # Might be an int.

    # Image.new wants (width, height) so swap them.
    image_size = image_config.base_shape[::-1]

    # Create the image, draw on the text, return it.
    image = Image.new('RGB', image_size)
    draw_text_grid(image, text)
    data = np.array(image)

    with block_timer("create_multi_scale_from_image", print_time=True):
        return create_multi_scale_from_image(data, image_config.tile_size)
