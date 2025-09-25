from typing import no_type_check

import numpy as np
from vispy.scene.visuals import Text

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

    # these magic numbers (1.2 and 1.3) are from trial and error
    return (bottom_right[0] - top_left[0]) * text.font_size, (
        bottom_right[1] - top_left[1]
    ) * text.font_size


# vendored from vispy/visuals/text/text.py, but removing all lines referring to context flushing,
# canvas and viewport which cause issues. We only need to calculate sizes anyways.
@no_type_check
def _text_to_vbo(text, font, anchor_x, anchor_y, lowres_size):
    """Convert text characters to VBO"""
    text_vtype = np.dtype(
        [('a_position', np.float32, 2), ('a_texcoord', np.float32, 2)]
    )
    vertices = np.zeros(len(text) * 4, dtype=text_vtype)
    prev = None
    width = height = ascender = descender = 0
    ratio, slop = 1.0 / font.ratio, font.slop
    x_off = -slop
    # Need to store the original viewport, because the font[char] will
    # trigger SDF rendering, which changes our viewport
    # todo: get rid of call to glGetParameter!

    # Also analyse chars with large ascender and descender, otherwise the
    # vertical alignment can be very inconsistent
    for char in 'hy':
        glyph = font[char]
        y0 = glyph['offset'][1] * ratio + slop
        y1 = y0 - glyph['size'][1]
        ascender = max(ascender, y0 - slop)
        descender = min(descender, y1 + slop)
        height = max(height, glyph['size'][1] - 2 * slop)

    # Get/set the fonts whitespace length and line height (size of this ok?)
    glyph = font[' ']
    spacewidth = glyph['advance'] * ratio
    lineheight = height * 1.5

    # Added escape sequences characters: {unicode:offset,...}
    #   ord('\a') = 7
    #   ord('\b') = 8
    #   ord('\f') = 12
    #   ord('\n') = 10  => linebreak
    #   ord('\r') = 13
    #   ord('\t') = 9   => tab, set equal 4 whitespaces?
    #   ord('\v') = 11  => vertical tab, set equal 4 linebreaks?
    # If text coordinate offset > 0 -> it applies to x-direction
    # If text coordinate offset < 0 -> it applies to y-direction
    esc_seq = {7: 0, 8: 0, 9: -4, 10: 1, 11: 4, 12: 0, 13: 0}

    # Keep track of y_offset to set lines at right position
    y_offset = 0

    # When a line break occur, record the vertices index value
    vi_marker = 0
    ii_offset = 0  # Offset since certain characters won't be drawn

    # The running tracker of characters vertex index
    vi = 0

    for ii, char in enumerate(text):
        if ord(char) in esc_seq:
            if esc_seq[ord(char)] < 0:
                # Add offset in x-direction
                x_off += abs(esc_seq[ord(char)]) * spacewidth
                width += abs(esc_seq[ord(char)]) * spacewidth
            elif esc_seq[ord(char)] > 0:
                # Add offset in y-direction and reset things in x-direction
                dx = dy = 0
                if anchor_x == 'right':
                    dx = -width
                elif anchor_x == 'center':
                    dx = -width / 2.0
                vertices['a_position'][vi_marker : vi + 4] += (dx, dy)
                vi_marker = vi + 4
                ii_offset -= 1
                # Reset variables that affects x-direction positioning
                x_off = -slop
                width = 0
                # Add offset in y-direction
                y_offset += esc_seq[ord(char)] * lineheight
        else:
            # For ordinary characters, normal procedure
            glyph = font[char]
            kerning = glyph['kerning'].get(prev, 0.0) * ratio
            x0 = x_off + glyph['offset'][0] * ratio + kerning
            y0 = glyph['offset'][1] * ratio + slop - y_offset
            x1 = x0 + glyph['size'][0]
            y1 = y0 - glyph['size'][1]
            u0, v0, u1, v1 = glyph['texcoords']
            position = [[x0, y0], [x0, y1], [x1, y1], [x1, y0]]
            texcoords = [[u0, v0], [u0, v1], [u1, v1], [u1, v0]]
            vi = (ii + ii_offset) * 4
            vertices['a_position'][vi : vi + 4] = position
            vertices['a_texcoord'][vi : vi + 4] = texcoords
            x_move = glyph['advance'] * ratio + kerning
            x_off += x_move
            ascender = max(ascender, y0 - slop)
            descender = min(descender, y1 + slop)
            width += x_move
            height = max(height, glyph['size'][1] - 2 * slop)
            prev = char

    dx = dy = 0
    if anchor_y == 'top':
        dy = -descender
    elif anchor_y in ('center', 'middle'):
        dy = (-descender - ascender) / 2
    elif anchor_y == 'bottom':
        dy = -ascender
    if anchor_x == 'right':
        dx = -width
    elif anchor_x == 'center':
        dx = -width / 2.0

    # If any linebreaks occured in text, we only want to translate characters
    # in the last line in text (those after the vi_marker)
    vertices['a_position'][0:vi_marker] += (0, dy)
    vertices['a_position'][vi_marker:] += (dx, dy)
    vertices['a_position'] /= lowres_size

    return vertices
