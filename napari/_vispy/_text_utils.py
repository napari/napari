from typing import Tuple

import numpy as np
from vispy.scene.visuals import Text


def update_text(
    text_values: np.ndarray,
    coords: np.ndarray,
    anchor: Tuple[str, str],
    rotation: float,
    color: np.ndarray,
    size: float,
    ndisplay: int,
    text_node: Text,
):
    """Update the vispy text node with the current text and display parameters.

    Parameters
    ----------
    text_values : np.ndarray
        The array of text strings to display.
    coords : np.ndarray
        The coordinates for each text element.
    anchor : Tuple[str, str]
        The name of the vispy anchor positions provided as (anchor_x, anchor_y).
        anchor_x should be one of: 'left', 'center', 'right'.
        anchor_y should be one of: 'top', 'center', 'middle', 'baseline', 'bottom'.
    rotation : float
        The rotation (degrees) of the text element around its anchor.
    color : np.ndarray
        The color of the text in an RGBA array.
    size : float
        The size of the font in points.
    ndisplay : int
        The number of dimensions displayed in the viewer.
    text_node : vispy.scene.visuals.Text
        The text node to be updated.
    """

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

    text_node.text = text_values
    text_node.pos = positions
    text_node.anchors = anchor
    text_node.rotation = rotation
    text_node.color = color
    text_node.font_size = size
