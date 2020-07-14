import numpy as np
from vispy.scene.visuals import Text


def update_text(
    text_values: np.ndarray,
    text_coords: np.ndarray,
    text_rotation: float,
    text_color: np.ndarray,
    text_size: float,
    ndisplay: int,
    text_node: Text,
):

    if ndisplay == 2:
        positions = np.flip(text_coords, axis=1)
    elif ndisplay == 3:
        raw_positions = np.flip(text_coords, axis=1)
        n_positions, position_dims = raw_positions.shape

        if position_dims < 3:
            padded_positions = np.zeros((n_positions, 3))
            padded_positions[:, 0:2] = raw_positions
            positions = padded_positions
        else:
            positions = raw_positions

    text_node.text = text_values
    text_node.pos = positions
    text_node.rotation = text_rotation
    text_node.color = text_color
    text_node.font_size = text_size
