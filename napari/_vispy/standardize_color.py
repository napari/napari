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
    color_switch = {
        str: handle_str,
        list: handle_array_like,
        tuple: handle_array_like,
        np.ndarray: handle_array_like,
        Color: handle_color,
        ColorArray: handle_color,
    }

    return color_switch[colortype](colors)


def handle_str(colors) -> np.ndarray:
    return Color(colors).rgba


def handle_array_like(colors) -> np.ndarray:
    return ColorArray(colors).rgba


def handle_color(colors) -> np.ndarray:
    return ColorArray(colors).rgba
