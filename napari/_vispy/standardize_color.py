import types

import numpy as np
from vispy.color import Color, ColorArray


def transform_color(colors) -> np.ndarray:
    """Receives the user-given colors - either edge or face -
    and transforms them into a numpy array that vispy can easily
    handle.

    The colors input can be a string, array-like, Color and
    ColorArray instances, or a mix of the above.
    """
    colortype = type(colors)
    return color_switch[colortype](colors)


def handle_str(colors) -> np.ndarray:
    """Creates an array from a color that was given as a string."""
    return Color(colors).rgba


def handle_array_like(colors) -> np.ndarray:
    if type(colors) is np.ndarray:
        if colors.ndim == 2 and colors.shape[0] == 1:
            colors = colors[0]
    try:
        color = ColorArray(colors).rgba
    except ValueError:
        return handle_nested_colors(colors)
    else:
        return color


def handle_color(colors) -> np.ndarray:
    return ColorArray(colors).rgba


def handle_generator(colors) -> np.ndarray:
    return handle_array_like(list(colors))


def handle_nested_colors(colors) -> np.ndarray:
    colors_as_rbga = np.ones((len(colors), 4), dtype=np.float32)
    for idx, color in enumerate(colors):
        colors_as_rbga[idx] = color_switch[type(color)](color)
    return colors_as_rbga


color_switch = {
    str: handle_str,
    list: handle_array_like,
    tuple: handle_array_like,
    types.GeneratorType: handle_generator,
    np.ndarray: handle_array_like,
    Color: handle_color,
    ColorArray: handle_color,
}
