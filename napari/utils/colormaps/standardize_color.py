"""This module contains functions that 'standardize' the color handling
of napari layers by supplying functions that are able to convert most
color representation the user had in mind into a single representation -
a numpy Nx4 array of float32 values - that is used across the codebase.

The main function of the module is "transform_color", which might call
a cascade of other, private, function in the module to do the hard work
of converting the input.

In general, we try to catch invalid color representations, warn the users
of their misbehaving and return a default white color array, since it seems
unreasonable to crash the entire napari session to mis-represented colors.
"""

import types
from typing import Iterable, Union, Dict, Any, Callable
import functools
import warnings

import numpy as np
from vispy.color import Color, ColorArray, get_color_dict, get_color_names
from vispy.color.color_array import _string_to_rgb


def transform_color(colors: Any) -> np.ndarray:
    """Receives the user-given colors and transforms them into an array of
    Nx4 np.float32 values, with N being the number of colors. The function
    (via its subfunctions, marked with _handle_X) is designed to parse all
    valid color representations a user might have and convert them properly.
    That being said, combinations of different color representation in the
    same list of colors is prohibited, and will error. This means that a list
    of ['red', np.array([1, 0, 0])] cannot be parsed and has to be manually
    pre-processed by the user before sent to this function.

    Parameters
    --------
    colors : string, array-like, Color and ColorArray instances, or a mix of the above.
        The color(s) to interpret and convert

    Returns
    ------
    colors : np.ndarray
        An instance of np.ndarray with a data type of float32, 4 columns in
        RGBA order and N rows, with N being the number of colors. The array
        will always be 2D even if a single color is passed.

    Raises
    -----
    ValueError, AttributeError, KeyError
        invalid inputs
    """
    colortype = type(colors)
    return _color_switch[colortype](colors)


@functools.lru_cache(maxsize=1024)
def _handle_str(color: str) -> np.ndarray:
    """Creates an array from a color that was given as a string."""
    # This line will stay here until vispy adds a "transparent" key
    # to their color dictionary. A PR was sent and approved, currently
    # waiting to be merged.
    if len(color) == 0:
        warnings.warn(
            "Empty string detected. Returning a black color instead."
        )
        return np.zeros((1, 4), dtype=np.float32)
    color = color.replace("transparent", "#00000000")
    as_arr = np.atleast_2d(_string_to_rgb(color)).astype(np.float32)
    if as_arr.shape[1] == 3:
        as_arr = np.column_stack([as_arr, np.float32(1.0)])
    return as_arr


def _handle_list_like(colors: Iterable) -> np.ndarray:
    """Handles all list-like containers of colors, using recursion. Numpy
    arrays are handled in _handle_array. Lists which are known to contain
    strings will be parsed with _handle_str_list_like.
    """
    try:
        # The following conversion works for most cases, and so it's expected
        # that most valid inputs will pass this .asarray() call
        # with ease. Those who don't are usually too cryptic to decipher. The
        # only exception is a list-like container with Vispy's Color and
        # ColorArray which is considered valid.
        color_array = np.atleast_2d(np.asarray(colors))
    except ValueError:
        if type(colors[0]) in (Color, ColorArray):
            color_array = np.vstack([c.rgba for c in colors])
            return color_array
        else:
            warnings.warn(
                "Coudln't convert input color array to a proper numpy array."
                " Please make sure that your input data is in a parsable format."
                " Converting input to a white color array."
            )
            return np.ones((max(len(colors), 1), 4), dtype=np.float32)

    # Happy path - converted to a float\integer array
    if color_array.dtype.kind in ['f', 'i']:
        return _handle_array(color_array)

    # User input was an iterable with strings
    if color_array.dtype.kind in ['U', 'O']:
        return _handle_str_list_like(color_array.ravel())


def _handle_vispy_color(colors: Union[Color, ColorArray]) -> np.ndarray:
    """Convert vispy's types to plain numpy arrays."""
    return np.atleast_2d(colors.rgba)


def _handle_generator(colors) -> np.ndarray:
    """Generators are converted to lists since we need to know their
    length to instantiate a proper array.
    """
    return _handle_list_like(list(colors))


def handle_nested_colors(colors) -> ColorArray:
    """In case of an array-like container holding colors, unpack it."""
    colors_as_rbga = np.ones((len(colors), 4), dtype=np.float32)
    for idx, color in enumerate(colors):
        colors_as_rbga[idx] = _color_switch[type(color)](color)
    return ColorArray(colors_as_rbga)


def _handle_array(colors: np.ndarray) -> np.ndarray:
    """Converts the given array into an array in the right format."""
    kind = colors.dtype.kind

    # Object arrays aren't handled by napari
    if kind == 'O':
        warnings.warn(
            "An object array was passed as the color input."
            " Please convert its datatype before sending it to napari."
            " Converting input to a white color array."
        )
        return np.ones((max(len(colors), 1), 4), dtype=np.float32)

    # An array of strings will be treated as a list if compatible
    elif kind == 'U':
        if colors.ndim == 1:
            return _handle_str_list_like(colors)
        else:
            warnings.warn(
                "String color arrays should be one-dimensional."
                " Converting input to a white color array."
            )
            return np.ones((len(colors), 4), dtype=np.float32)

    # Test the dimensionality of the input array

    # Empty color array can be a way for the user to signal
    # that it wants the "default" colors of napari. We return
    # a single white color.
    if colors.shape[-1] == 0:
        warnings.warn(
            "Given color input is empty. Converting input to"
            " a white color array."
        )
        return np.ones((1, 4), dtype=np.float32)

    colors = np.atleast_2d(colors)

    # Arrays with more than two dimensions don't have a clear
    # conversion method to a color array and thus raise an error.
    if colors.ndim > 2:
        raise ValueError(
            "Given colors input should contain one or two dimensions."
            f" Received array with {colors.ndim} dimensions."
        )

    # User provided a list of numbers as color input. This input
    # cannot be coerced into something understandable and thus
    # will return an error.
    if colors.shape[0] == 1 and colors.shape[1] not in {3, 4}:
        raise ValueError(
            "Given color array has an unsupported format."
            f" Received the following array:\n{colors}\n"
            "A proper color array should have 3-4 columns"
            " with a row per data entry."
        )

    # The user gave a list of colors, but it contains a wrong number
    # of columns. This check will also drop Nx1 (2D) arrays, since
    # numpy has vectors, and representing colors in this way
    # (column vector-like) is redundant. However, this results in a
    # warning and not a ValueError since we know the number of colors
    # in this dataset, meaning we can save the napari session by
    # rendering the data in white, which better than crashing.
    if not 3 <= colors.shape[1] <= 4:
        warnings.warn(
            "Given colors input should contain three or four columns."
            f" Received array with {colors.shape[1]} columns."
            " Converting input to a white color array."
        )
        return np.ones((len(colors), 4), dtype=np.float32)

    # Arrays with floats and ints can be safely converted to the proper format
    if kind in ['f', 'i', 'u']:
        return _convert_array_to_correct_format(colors)

    else:
        raise ValueError(f"Data type of array ({colors.dtype}) not supported.")


def _convert_array_to_correct_format(colors: np.ndarray) -> np.ndarray:
    """This function deals with arrays which are already 'well-behaved',
    i.e have (almost) the correct number of columns and are able to represent
    colors correctly, and then it makes sure that the array indeed has exactly
    four columns and that its values are normalized between 0 and 1, with a
    data type of float32.
    """
    if colors.shape[1] == 3:
        colors = np.column_stack(
            [colors, np.ones(len(colors), dtype=np.float32)]
        )

    if colors.min() < 0:
        raise ValueError("Colors input had negative values.")

    if colors.max() > 1:
        warnings.warn(
            "Colors with values larger than one detected. napari"
            " will normalize these colors for you. If you'd like to convert these"
            " yourself, please use the proper method from scikit-image.color."
        )
        colors = _normalize_color_array(colors)
    return np.atleast_2d(np.asarray(colors, dtype=np.float32))


def _handle_str_list_like(colors: Iterable) -> np.ndarray:
    """Handles lists or arrays filled with strings."""
    color_array = np.empty((len(colors), 4), dtype=np.float32)
    for idx, c in enumerate(colors):
        try:
            color_array[idx, :] = _color_switch[type(c)](c)
        except (ValueError, TypeError, KeyError):
            raise ValueError(f"Invalid color found: {c} at index {idx}.")
    return color_array


def _handle_none(color) -> np.ndarray:
    """A None color is assumed to be black."""
    return np.zeros((1, 4), dtype=np.float32)


def _normalize_color_array(colors: np.ndarray) -> np.ndarray:
    """Normalizes all array values in the range [0, 1].

    The added complexity here stems from the fact that if a row in the given
    array contains four identical value a simple normalization will raise a
    division by zero exception.
    """
    colors = colors.astype(np.float32)
    out_of_bounds_idx = np.unique(np.where((colors > 1) | (colors < 0))[0])
    out_of_bounds = colors[out_of_bounds_idx]
    norm = np.linalg.norm(out_of_bounds, np.inf, axis=1)
    out_of_bounds = out_of_bounds / norm[:, np.newaxis]
    colors[out_of_bounds_idx] = out_of_bounds
    return colors.astype(np.float32)


_color_switch: Dict[Any, Callable] = {
    str: _handle_str,
    np.str_: _handle_str,
    list: _handle_list_like,
    tuple: _handle_list_like,
    types.GeneratorType: _handle_generator,
    np.ndarray: _handle_array,
    Color: _handle_vispy_color,
    ColorArray: _handle_vispy_color,
    type(None): _handle_none,
}


def _convert_color_hex_to_name():
    """Create a dictionary converting hexadecimal RGB colors into their
    'official' name.

    Returns
    -----
    hex_to_rgb : dict
        Mapping from hexadecimal RGB ('#ff0000') to name ('red').
    """
    colordict = get_color_dict()
    return {f"{v.lower()}ff": k for k, v in colordict.items()}


def get_color_namelist():
    """A simple wrapper around vispy's get_color_names. It also adds the
    'transparent' color to that list. Once https://github.com/vispy/vispy/pull/1794
    is merged this function is no longer necessary.
    """
    names = get_color_names()
    names.append("transparent")
    return names


hex_to_name = _convert_color_hex_to_name()


def rgb_to_hsv():
    pass


def hsv_to_rgb():
    pass


def _check_color_dim(val):
    """Ensure val is Nx(n_col), usually Nx3"""
    val = np.atleast_2d(val)
    if val.shape[1] not in (3, 4):
        raise RuntimeError('Value must have second dimension of size 3 or 4')
    if val.shape[1] == 3:
        val = np.column_stack([val, np.float32(1.0)])
    return val


def rgb_to_hex(rgbs):
    """Convert rgb to hex triplet. Taken from vispy with slight
    modifications."""
    rgbs = _check_color_dim(rgbs)
    return np.array(
        [
            f'#{"%02x" * 4}' % tuple((255 * rgb).astype(np.uint8))
            for rgb in rgbs
        ],
        '|U9',
    )
