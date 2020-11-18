"""Create test images

This is a throw-away file for creating a test image for octree rendering
development. If we keep the in-napari creation of test images long term,
we'll probably want a better system for creating them.

Also, long term we probably do not want to use PIL at all, since it's
not a desirable dependency.
"""
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ....layers.image.experimental import (
    SliceConfig,
    create_multi_scale_from_image,
)
from ....utils.perf import block_timer


class TextWriter:
    """Write text into a PIL image."""

    def __init__(self, draw: ImageDraw.Draw, font_str: str, color: str):
        self.draw = draw
        try:
            self.font = ImageFont.truetype(font_str, size=74)
        except OSError:
            self.font = ImageFont.load_default()
        self.color = color

    def get_text_size(self, text: str) -> Tuple[int, int]:
        """Get size of the given text."""
        return self.font.getsize(text)

    def write_text(self, pos: Tuple[int, int], text: str) -> None:
        """Write the text into the image."""
        self.draw.text(pos, text, fill=self.color, font=self.font)


def draw_text_grid(image, text: str) -> None:
    """Draw some text into the given image in a grid.

    Parameters
    ----------
    test : str
        The text to draw. For example a slice index like "3".
    """
    color = 'rgb(255, 255, 255)'  # white
    width, height = image.size
    draw = ImageDraw.Draw(image)

    writer = TextWriter(draw, 'Arial Black.ttf', color)
    (text_width, text_height) = writer.get_text_size(text)

    text_size = 100  # hack, approx guess with some padding

    rows, cols = int(height / text_size), int(width / text_size)

    for row in range(rows + 1):
        for col in range(cols + 1):
            pos = [
                (col / cols) * image.width - text_width / 2,  # x
                (row / rows) * image.height - text_height / 2,  # y
            ]

            writer.write_text(pos, text)

    draw.rectangle([0, 0, image.width, image.height], outline=color, width=5)


def create_test_image(text, slice_config: SliceConfig) -> np.ndarray:
    """Create a test image for testing tiled rendering.

    The test image just has digits all over it. The digits will typically
    be used to show the slice number like "0" or "42".

    slice_config: SliceConfig
        The base image shape and other details.
    """
    text = str(text)  # Might be an int.

    # Image.new wants (width, height) so swap them.
    image_size = slice_config.image_shape[::-1]

    # Create the image, draw on the text, return it.
    image = Image.new('RGB', image_size)
    draw_text_grid(image, text)
    return np.array(image)


def create_test_image_multi(text, slice_config: SliceConfig) -> np.ndarray:
    """Create a multiscale test image for testing tiled rendering.

    The test image is blank with digits all over it. The digits will
    typically be used to show the slice number.

    slice_config: SliceConfig
        The base image shape and other details.
    """
    text = str(text)  # Convert in case it's an int.

    # Image.new wants (width, height) so swap them.
    image_size = slice_config.base_shape[::-1]

    # Create the image, draw on the text, return it.
    image = Image.new('RGB', image_size)
    draw_text_grid(image, text)
    data = np.array(image)

    with block_timer("create_multi_scale_from_image", print_time=True):
        # This is create all the octree layers, each coarser one
        # downsampled from the previous level.
        return create_multi_scale_from_image(data, slice_config.tile_size)
