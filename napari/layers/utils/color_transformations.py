"""This file contains functions which are designed to assist Layer objects transform,
normalize and broadcast the color inputs they receive into a more standardized format -
a numpy array with N rows, N being the number of data points, and a dtype of np.float32.

"""
from itertools import cycle
from typing import Union, List, Tuple, AnyStr
import warnings

from vispy.color import Color, ColorArray
import numpy as np

from ...utils.colormaps.standardize_color import transform_color


# All parsable input datatypes that a user can provide
ColorType = Union[List, Tuple, np.ndarray, AnyStr, Color, ColorArray]


def transform_color_with_defaults(
    num_entries: int, colors: ColorType, elem_name: str, default: str
) -> np.ndarray:
    """Helper method to return an Nx4 np.array from an arbitrary user input.

    Parameters
    ----------
    num_entries : int
        The number of data elements in the layer
    colors : ColorType
        The wanted colors for each of the data points
    elem_name : str
        Whether we're trying to set the face color or edge color of the layer
    default : str
        The default color for that element in the layer

    Returns
    -------
    transformed : np.ndarray
        Nx4 numpy array with a dtype of np.float32
    """
    try:
        transformed = transform_color(colors)
    except (AttributeError, ValueError, KeyError):
        warnings.warn(
            f"The provided {elem_name} parameter contained illegal values, "
            f"reseting all {elem_name} values to {default}."
        )
        transformed = transform_color(default)
    else:
        if (len(transformed) != 1) and (len(transformed) != num_entries):
            warnings.warn(
                f"The provided {elem_name} parameter has {len(colors)} entries, "
                f"while the data contains {num_entries} entries. Setting {elem_name} to {default}."
            )
            transformed = transform_color(default)
    return transformed


def transform_color_cycle(
    color_cycle: Union[ColorType, cycle], elem_name: str, default: str
) -> cycle:
    """Helper method to return an Nx4 np.array from an arbitrary user input.

    Parameters
    ----------
    colors : ColorType, cycle
        The desired colors for each of the data points
    elem_name : str
        Whether we're trying to set the face color or edge color of the layer
    default : str
        The default color for that element in the layer

    Returns
    -------
    transformed : cycle
        cycle of Nx4 numpy arrays with a dtype of np.float32
    """

    if isinstance(color_cycle, cycle):
        transformed = color_cycle
    else:
        transformed_color_cycle = transform_color_with_defaults(
            num_entries=len(color_cycle),
            colors=color_cycle,
            elem_name=elem_name,
            default=default,
        )
        transformed = cycle(transformed_color_cycle)

    return transformed


def normalize_and_broadcast_colors(
    num_entries: int, colors: ColorType
) -> np.ndarray:
    """Takes an input color array and forces into being the length of ``data``.

    Used when a single color is supplied for many input objects, but we need
    Layer.current_face_color or Layer.current_edge_color to have the shape of
    the actual data.

    Note: This function can't robustly parse user input, and thus should
    always be used on the output of ``transform_color_with_defaults``.

    Parameters
    ----------
    num_entries : int
        The number of data elements in the layer
    color : ColorType
        The user's input after being normalized by transform_color_with_defaults

    Returns
    -------
    tiled : np.ndarray
        A tiled version (if needed) of the original input
    """
    # len == 0 data is handled somewhere else
    if (len(colors) == num_entries) or (num_entries == 0):
        return np.asarray(colors)
    # If the user has supplied a list of colors, but its length doesn't
    # match the length of the data, we warn them and return a single
    # color for all inputs
    if len(colors) != 1:
        warnings.warn(
            f"The number of supplied colors mismatch the number of given"
            f" data points. Length of data is {num_entries}, while the number of colors"
            f" is {len(colors)}. Color for all points is reset to white."
        )
        tiled = np.ones((num_entries, 4), dtype=np.float32)
        return tiled
    # All that's left is to deal with length=1 color inputs
    tiled = np.tile(colors.ravel(), (num_entries, 1))
    return tiled
