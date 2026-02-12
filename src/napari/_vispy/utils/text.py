from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import TYPE_CHECKING

import numpy as np
from PIL.ImageFont import FreeTypeFont

from napari.layers import Points, Shapes
from napari.layers.utils.string_encoding import ConstantStringEncoding

if TYPE_CHECKING:
    from vispy.visuals.text import Text


FONT_DIR = resources.files('napari') / 'resources' / 'fonts' / 'AlataPlus'
FONT_FILE = FONT_DIR / 'AlataPlus-Regular.ttf'


def update_text(
    *,
    node: Text,
    layer: Points | Shapes,
):
    """Update the vispy text node with a layer's text parameters.

    Parameters
    ----------
    node : vispy.scene.visuals.Text
        The text node to be updated.
    layer : Union[Points, Shapes]
        A layer with text.
    """

    ndisplay = layer._slice_input.ndisplay

    # Vispy always needs non-empty values and coordinates, so if a layer
    # effectively has no visible text then return single dummy data.
    # This also acts as a minor optimization.
    if _has_visible_text(layer):
        text_values = layer._view_text
        colors = layer._view_text_color
        coords, anchor_x, anchor_y = layer._view_text_coords
    else:
        text_values = np.array([''])
        colors = np.zeros((4,), np.float32)
        coords = np.zeros((1, ndisplay))
        anchor_x = 'center'
        anchor_y = 'center'

    # Vispy wants (x, y) positions instead of (row, column) coordinates.
    if ndisplay == 2:
        positions = np.flip(coords, axis=1)
    elif ndisplay == 3:
        raw_positions = np.flip(coords, axis=1)
        n_positions, position_dims = raw_positions.shape

        if position_dims < 3:
            padded_positions = np.zeros((n_positions, 3))
            padded_positions[:, 0:2] = raw_positions
            positions = padded_positions
        else:
            positions = raw_positions

    node.text = text_values
    node.pos = positions
    node.anchors = (anchor_x, anchor_y)

    text_manager = layer.text
    node.rotation = text_manager.rotation
    node.color = colors

    node.font_size = text_manager._get_scaled_size(layer.scale_factor)


def _has_visible_text(layer: Points | Shapes) -> bool:
    text = layer.text
    if not text.visible:
        return False
    if (
        isinstance(text.string, ConstantStringEncoding)
        and text.string.constant == ''
    ):
        return False
    return len(layer._indices_view) != 0


@lru_cache
def _load_font(size=10):
    return FreeTypeFont(str(FONT_FILE), size=size)


def get_text_width_height(text: Text) -> tuple[float, float]:
    """Get the width and height of a vispy text visual in screen pixels.

    If display scaling is not 1 (e.g. hidpi), this is already accounted for
    by vispy.
    """
    if isinstance(text.text, str):
        strings = [text.text]
    elif isinstance(text.text, list):
        strings = text.text
    else:
        raise TypeError('Text should either be a string or a list of strings')

    font = _load_font(size=text.font_size)
    font_height = sum(font.getmetrics()) * 0.81

    height = 0
    width = 0

    for string in strings:
        if string == '':
            continue

        for lineno, line in enumerate(string.split('\n')):
            width = max(width, font.getlength(line))
            height_lines = font_height * (lineno + 1)
            height_line_spacing = font_height * (text.line_height - 1) * lineno
            height = max(height, height_lines + height_line_spacing)

    return width * text.dpi_ratio, height * text.dpi_ratio


def register_napari_fonts() -> None:
    from vispy.util.fonts import register_vispy_font

    register_vispy_font(FONT_DIR, 'AlataPlus', False, False)
