import types

import numpy as np
from vispy.color import Color, ColorArray


def transform_color(colors) -> ColorArray:
    """Receives the user-given colors and transforms them into a ColorArray.

    Parameters
    --------
    colors : string, array-like, Color and ColorArray instances, or a mix of the above.
        The color to interpret

    Raises
    -----
    ValueError, AttributeError, KeyError
        invalid inputs
    """
    colortype = type(colors)
    return color_switch[colortype](colors)


def handle_str(colors) -> ColorArray:
    """Creates an array from a color that was given as a string."""
    return ColorArray(colors)


def handle_array_like(colors) -> ColorArray:
    """Handles all array-like containers of colors, using recursion."""
    if type(colors) is np.ndarray:
        if colors.ndim == 2 and colors.shape[0] == 1:
            colors = colors[0]
    try:
        color = ColorArray(colors)
    except ValueError:
        return handle_nested_colors(colors)
    else:
        return color


def handle_color(colors) -> ColorArray:
    return ColorArray(colors)


def handle_generator(colors) -> ColorArray:
    return handle_array_like(list(colors))


def handle_nested_colors(colors) -> ColorArray:
    """In case of an array-like container holding colors, unpack it."""
    colors_as_rbga = np.ones((len(colors), 4), dtype=np.float32)
    for idx, color in enumerate(colors):
        colors_as_rbga[idx] = color_switch[type(color)](color).rgba
    return ColorArray(colors_as_rbga)


color_switch = {
    str: handle_str,
    np.str_: handle_str,
    list: handle_array_like,
    tuple: handle_array_like,
    types.GeneratorType: handle_generator,
    np.ndarray: handle_array_like,
    Color: handle_color,
    ColorArray: handle_color,
}
