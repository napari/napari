from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtGui import QFontMetricsF, QGuiApplication

from napari._vispy.utils.qt_font import SDF_FONT_SIZE, make_qfont
from napari.layers import Points, Shapes
from napari.layers.utils.string_encoding import ConstantStringEncoding

if TYPE_CHECKING:
    from napari._vispy.visuals.text import Text


def update_text(
    *,
    node: Text,
    layer: Points | Shapes,
) -> None:
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
    node.pos = positions  # pyrefly: ignore [unbound-name]
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
    return len(layer._view_indices) != 0


@lru_cache(maxsize=128)
def _get_qt_font_metrics(
    face: str, size: int, bold: bool = False, italic: bool = False
):
    """Get cached Qt font metrics for the given font properties.

    Parameters
    ----------
    face : str
        Font face name.
    size : int
        Font size in points.
    bold : bool, optional
        Whether the font is bold.
    italic : bool, optional
        Whether the font is italic.

    Returns
    -------
    QFontMetricsF
        Qt font metrics object.
    """
    return QFontMetricsF(make_qfont(face, size, bold=bold, italic=italic))


def _get_scaled_metrics(text: Text) -> tuple[QFontMetricsF, float]:
    """Get high-res Qt font metrics for a text visual and their scale factor.

    Glyphs are rendered from a high-res font scaled down (see QtTextureFont),
    so measure at that same size and scale the result: at small point sizes
    Qt rounds each glyph advance up, which made measured text wider than the
    rendered one.
    """
    face = (
        text.face if hasattr(text, 'face') else QGuiApplication.font().family()
    )
    bold = text.bold if hasattr(text, 'bold') else False
    italic = text.italic if hasattr(text, 'italic') else False

    metrics = _get_qt_font_metrics(face, SDF_FONT_SIZE, bold, italic)
    return metrics, text.font_size / SDF_FONT_SIZE


def get_text_line_height(text: Text) -> float:
    """Get the line height of a vispy text visual, like get_text_width_height."""
    metrics, scale = _get_scaled_metrics(text)
    return metrics.height() * scale


def get_text_width_height(text: Text) -> tuple[float, float]:
    """Get the width and height of a vispy text visual in screen pixels.

    If display scaling is not 1 (e.g. hidpi), this is already accounted for
    by vispy.
    """
    if isinstance(text.text, str):
        string = text.text
    elif isinstance(text.text, list):
        string = '\n'.join(text.text)
    else:
        raise TypeError('Text should either be a string or a list of strings')

    metrics, scale = _get_scaled_metrics(text)
    size = metrics.size(0, string)

    return size.width() * scale, size.height() * scale
