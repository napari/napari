"""Create test images

This is a throw-away file for creating a test image for octree rendering
development. If we keep test images in the product long term we'll
have a nicer way to generate them.

Long term we probably do not want to use PIL for example.
"""
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_text_grid(image, text: str, grid_shape: Tuple[int, int]) -> None:
    """Draw some text into the given image in a grid.

    Parameters
    ----------
    test : str
        The text to draw. For example a slice index like "3".
    grid_shape : Tuple[int, int]
        Draw the text in a grid of is [height, weight] shape.
    """

    try:
        font = ImageFont.truetype('Arial Black.ttf', size=74)
    except OSError:
        font = ImageFont.load_default()
    (text_width, text_height) = font.getsize(text)

    color = 'rgb(255, 255, 255)'  # white
    draw = ImageDraw.Draw(image)

    rows, cols = grid_shape[0], grid_shape[1]

    for row in range(rows + 1):
        for col in range(cols + 1):
            y = (row / rows) * image.height - text_height / 2
            x = (col / cols) * image.width - text_width / 2

            draw.text((x, y), text, fill=color, font=font)
    draw.rectangle([0, 0, image.width, image.height], outline=color, width=5)


def create_test_image(
    text,
    digit_shape: Tuple[int, int],
    image_shape: Tuple[int, int] = (1024, 1024),
) -> np.ndarray:
    """Create a test image for testing tiled rendering.

    The test image just has digits all over it. The digits will typically
    be used to show the slice number.

    shape: Tuple[int, int]
        The [height, width] shape of the image.
    """
    text = str(text)  # Might be an int.

    # Image.new wants (width, height) so swap them.
    image_size = (
        image_shape[1],
        image_shape[0],
    )

    # Create the image, draw on the text, return it.
    image = Image.new('RGB', image_size)
    draw_text_grid(image, text, digit_shape)
    return np.array(image)
