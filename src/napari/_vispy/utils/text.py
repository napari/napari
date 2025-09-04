import numpy as np
from vispy.scene.visuals import Text

# Use vispy private function for convenience when calculating bounding box. Consider its behavior when upgrading vispy version.
from vispy.visuals.text.text import _text_to_vbo

from napari.layers import Points, Shapes
from napari.layers.utils.string_encoding import ConstantStringEncoding


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


def get_text_width_height(text: Text) -> tuple[float, float]:
    """Get the screen space width and height of a vispy text visual."""
    if isinstance(text.text, str):
        strings = [text.text]
    elif isinstance(text.text, list):
        strings = text.text
    else:
        raise TypeError('Text should either be a string or a list of strings')

    top_left_corners = []
    bottom_right_corners = []
    for string in strings:
        if string == '':
            continue

        # this is a private vispy function that calculates the Vertex Buffer Object
        # for the text to be sent to the shaders. This normallly only happens on
        # the fly when rendering, but we need it here because it allows us to
        # calculate the bounding box of the text as it would be when rendered
        buffer = _text_to_vbo(
            string, text._font, *text._anchors, text._font._lowres_size
        )

        pos = buffer['a_position']
        top_left_corners.append(pos.min(axis=0))
        bottom_right_corners.append(pos.max(axis=0))

    top_left = np.min(top_left_corners, axis=0) if top_left_corners else (0, 0)
    bottom_right = (
        np.max(bottom_right_corners, axis=0)
        if bottom_right_corners
        else (0, 0)
    )

    font_size = get_text_font_size(text)

    # these magic numbers (1.2 and 1.3) are from trial and error
    return (bottom_right[0] - top_left[0]) * font_size * 1.3, (
        bottom_right[1] - top_left[1]
    ) * font_size * 1.2


def get_text_font_size(text: Text) -> float:
    """Get the logical font size of a text visual, rescaled by dpi."""
    # use 96 as the vispy reference dpi for historical reasons
    dpi_scale_factor = 96 / text.transforms.dpi if text.transforms.dpi else 1

    return text.font_size * dpi_scale_factor
