import numpy as np


def update_text(layer, text_node):
    if len(layer._indices_view) == 0:
        text_coords = np.zeros((1, layer.dims.ndisplay))
        text = []
    else:
        text_coords = layer._view_text_coords
        text = layer._view_text

    if layer.dims.ndisplay == 2:
        positions = np.flip(text_coords, axis=1)
    elif layer.dims.ndisplay == 3:
        raw_positions = np.flip(text_coords, axis=1)
        n_positions, position_dims = raw_positions.shape

        if position_dims < 3:
            padded_positions = np.zeros((n_positions, 3))
            padded_positions[:, 0:2] = raw_positions
            positions = padded_positions
        else:
            positions = raw_positions

    text_node.text = text
    text_node.pos = positions
    text_node.rotation = layer._text.rotation
    text_node.color = layer._text.color
    text_node.font_size = layer._text.size
