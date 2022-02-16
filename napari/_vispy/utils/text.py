from typing import Union

import numpy as np
from vispy.scene.visuals import Text

from napari.layers import Points, Shapes
from napari.layers.utils.string_encoding import ConstantStringEncoding


def update_text_node(
    *,
    text_node: Text,
    layer: Union[Shapes, Points],
):
    """Update the vispy text node with a layer's text parameters.

    Parameters
    ----------
    text_node : vispy.scene.visuals.Text
        The text node to be updated.
    layer : Layer
        A layer with text.
    """

    ndisplay = layer._ndisplay

    # Vispy always needs non-empty values and coordinates, so if a layer
    # effectively has no visible text then return single dummy data.
    # This also acts as a minor optimization.
    if _has_visible_text(layer):
        text_values = layer._view_text
        coords, anchor_x, anchor_y = layer._view_text_coords
    else:
        text_values = np.array([''])
        coords, anchor_x, anchor_y = (
            np.zeros((1, ndisplay)),
            'center',
            'center',
        )

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

    text_node.text = text_values
    text_node.pos = positions
    text_node.anchors = (anchor_x, anchor_y)

    text_manager = layer.text
    text_node.rotation = text_manager.rotation
    text_node.color = text_manager.color
    text_node.font_size = text_manager.size


def _has_visible_text(layer: Union[Points, Shapes]) -> bool:
    text = layer.text
    if not text.visible:
        return False
    if len(layer._indices_view) == 0:
        return False
    elif (
        isinstance(text.string, ConstantStringEncoding)
        and text.string.constant == ''
    ):
        return False
    return True
