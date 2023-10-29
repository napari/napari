"""This file contains functions which are designed to assist Layer objects transform,
normalize and broadcast the color inputs they receive into a more standardized format -
a numpy array with N rows, N being the number of data points, and a dtype of np.float32.

"""
import warnings
from itertools import cycle
from typing import Tuple

import numpy as np

from napari.utils.colormaps.colormap_utils import ColorType
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.translations import trans


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
        Element we're trying to set the color, for example, `face_color` or `track_colors`.
        This is used to provide context to user warnings.
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
            trans._(
                "The provided {elem_name} parameter contained illegal values, resetting all {elem_name} values to {default}.",
                deferred=True,
                elem_name=elem_name,
                default=default,
            )
        )
        transformed = transform_color(default)
    else:
        if (len(transformed) != 1) and (len(transformed) != num_entries):
            warnings.warn(
                trans._(
                    "The provided {elem_name} parameter has {length} entries, while the data contains {num_entries} entries. Setting {elem_name} to {default}.",
                    deferred=True,
                    elem_name=elem_name,
                    length=len(colors),
                    num_entries=num_entries,
                    default=default,
                )
            )
            transformed = transform_color(default)
    return transformed


def transform_color_cycle(
    color_cycle: ColorType, elem_name: str, default: str
) -> Tuple["cycle[np.ndarray]", np.ndarray]:
    """Helper method to return an Nx4 np.array from an arbitrary user input.

    Parameters
    ----------
    color_cycle : ColorType
        The desired colors for each of the data points
    elem_name : str
        Whether we're trying to set the face color or edge color of the layer
    default : str
        The default color for that element in the layer

    Returns
    -------
    transformed_color_cycle : cycle
        cycle of shape (4,) numpy arrays with a dtype of np.float32
    transformed_colors : np.ndarray
        input array of colors transformed to RGBA
    """
    transformed_colors = transform_color_with_defaults(
        num_entries=len(color_cycle),
        colors=color_cycle,
        elem_name=elem_name,
        default=default,
    )
    transformed_color_cycle = cycle(transformed_colors)

    return transformed_color_cycle, transformed_colors


def normalize_and_broadcast_colors(
    num_entries: int, colors: np.ndarray
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
    colors : np.ndarray
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
            trans._(
                "The number of supplied colors mismatch the number of given data points. Length of data is {num_entries}, while the number of colors is {length}. Color for all points is reset to white.",
                deferred=True,
                num_entries=num_entries,
                length=len(colors),
            )
        )
        tiled = np.ones((num_entries, 4), dtype=np.float32)
        return tiled
    # All that's left is to deal with length=1 color inputs
    tiled = np.tile(colors.ravel(), (num_entries, 1))
    return tiled
