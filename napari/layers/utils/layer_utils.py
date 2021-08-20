from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import dask
import numpy as np

from ...utils.action_manager import action_manager
from ...utils.events.custom_types import Array
from ...utils.transforms import Affine
from ...utils.translations import trans

if TYPE_CHECKING:
    from pandas import DataFrame


def register_layer_action(keymapprovider, description: str, shortcuts=None):
    """
    Convenient decorator to register an action with the current Layers

    It will use the function name as the action name. We force the description
    to be given instead of function docstring for translation purpose.


    Parameters
    ----------
    keymapprovider : KeymapProvider
        class on which to register the keybindings – this will typically be
        the instance in focus that will handle the keyboard shortcut.
    description : str
        The description of the action, this will typically be translated and
        will be what will be used in tooltips.
    shortcuts : str | List[str]
        Shortcut to bind by default to the action we are registering.

    Returns
    -------
    function:
        Actual decorator to apply to a function. Given decorator returns the
        function unmodified to allow decorator stacking.

    """

    def _inner(func):
        nonlocal shortcuts
        name = 'napari:' + func.__name__
        action_manager.register_action(
            name=name,
            command=func,
            description=description,
            keymapprovider=keymapprovider,
        )
        if shortcuts:
            if isinstance(shortcuts, str):
                shortcuts = [shortcuts]

            for shortcut in shortcuts:
                action_manager.bind_shortcut(name, shortcut)
        return func

    return _inner


def calc_data_range(data, rgb=False):
    """Calculate range of data values. If all values are equal return [0, 1].

    Parameters
    ----------
    data : array
        Data to calculate range of values over.
    rgb : bool
        Flag if data is rgb.

    Returns
    -------
    values : list of float
        Range of values.

    Notes
    -----
    If the data type is uint8, no calculation is performed, and 0-255 is
    returned.
    """
    if data.dtype == np.uint8:
        return [0, 255]
    if np.prod(data.shape) > 1e7:
        # If data is very large take the average of the top, bottom, and
        # middle slices
        offset = 2 + int(rgb)
        bottom_plane_idx = (0,) * (data.ndim - offset)
        middle_plane_idx = tuple(s // 2 for s in data.shape[:-offset])
        top_plane_idx = tuple(s - 1 for s in data.shape[:-offset])
        idxs = [bottom_plane_idx, middle_plane_idx, top_plane_idx]
        # If each plane is also very large, look only at a subset of the image
        if (
            np.prod(data.shape[-offset:]) > 1e7
            and data.shape[-offset] > 64
            and data.shape[-offset + 1] > 64
        ):
            # Find a centeral patch of the image to take
            center = [int(s // 2) for s in data.shape[-offset:]]
            cental_slice = tuple(slice(c - 31, c + 31) for c in center[:2])
            reduced_data = [
                [np.max(data[idx + cental_slice]) for idx in idxs],
                [np.min(data[idx + cental_slice]) for idx in idxs],
            ]
        else:
            reduced_data = [
                [np.max(data[idx]) for idx in idxs],
                [np.min(data[idx]) for idx in idxs],
            ]
        # compute everything in one go
        reduced_data = dask.compute(*reduced_data)
    else:
        reduced_data = data

    min_val = np.min(reduced_data)
    max_val = np.max(reduced_data)

    if min_val == max_val:
        min_val = 0
        max_val = 1
    return [float(min_val), float(max_val)]


def segment_normal(a, b, p=(0, 0, 1)):
    """Determines the unit normal of the vector from a to b.

    Parameters
    ----------
    a : np.ndarray
        Length 2 array of first point or Nx2 array of points
    b : np.ndarray
        Length 2 array of second point or Nx2 array of points
    p : 3-tuple, optional
        orthogonal vector for segment calculation in 3D.

    Returns
    -------
    unit_norm : np.ndarray
        Length the unit normal of the vector from a to b. If a == b,
        then returns [0, 0] or Nx2 array of vectors
    """
    d = b - a

    if d.ndim == 1:
        if len(d) == 2:
            normal = np.array([d[1], -d[0]])
        else:
            normal = np.cross(d, p)
        norm = np.linalg.norm(normal)
        if norm == 0:
            norm = 1
    else:
        if d.shape[1] == 2:
            normal = np.stack([d[:, 1], -d[:, 0]], axis=0).transpose(1, 0)
        else:
            normal = np.cross(d, p)

        norm = np.linalg.norm(normal, axis=1, keepdims=True)
        ind = norm == 0
        norm[ind] = 1
    unit_norm = normal / norm

    return unit_norm


def convert_to_uint8(data: np.ndarray) -> np.ndarray:
    """
    Convert array content to uint8.

    If all negative values are changed on 0.

    If values are integer and bellow 256 it is simple casting otherwise maximum value for this data type is picked
    and values are scaled by 255/maximum type value.

    Binary images ar converted to [0,255] images.

    float images are multiply by 255 and then casted to uint8.

    Based on skimage.util.dtype.convert but limited to output type uint8
    """
    out_dtype = np.dtype(np.uint8)
    out_max = np.iinfo(out_dtype).max
    if data.dtype == out_dtype:
        return data
    in_kind = data.dtype.kind
    if in_kind == "b":
        return data.astype(out_dtype) * 255
    if in_kind == "f":
        image_out = np.multiply(data, out_max, dtype=data.dtype)
        np.rint(image_out, out=image_out)
        np.clip(image_out, 0, out_max, out=image_out)
        return image_out.astype(out_dtype)

    if in_kind in "ui":
        if in_kind == "u":
            if data.max() < out_max:
                return data.astype(out_dtype)
            return np.right_shift(data, (data.dtype.itemsize - 1) * 8).astype(
                out_dtype
            )
        else:
            np.maximum(data, 0, out=data, dtype=data.dtype)
            if data.dtype == np.int8:
                return (data * 2).astype(np.uint8)
            if data.max() < out_max:
                return data.astype(out_dtype)
            return np.right_shift(
                data, (data.dtype.itemsize - 1) * 8 - 1
            ).astype(out_dtype)


def prepare_properties(
    properties: Optional[Union[Dict[str, Array], DataFrame]],
    choices: Optional[Dict[str, Array]] = None,
    num_data: int = 0,
    save_choices: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Prepare properties and choices into standard forms.

    Parameters
    ----------
    properties : dict[str, Array] or DataFrame
        The property values.
    choices : dict[str, Array]
        The property value choices.
    num_data : int
        The length of data that the properties represent (e.g. number of points).
    save_choices : bool
        If true, always return all of the properties in choices.

    Returns
    -------
    properties: dict[str, np.ndarray]
        A dictionary where the key is the property name and the value
        is an ndarray of property values.
    choices: dict[str, np.ndarray]
        A dictionary where the key is the property name and the value
        is an ndarray of unique property value choices.
    """
    # If there is no data, non-empty properties represent choices as a deprecated behavior.
    if num_data == 0 and properties:
        warnings.warn(
            trans._(
                "Property choices should be passed as property_choices, not properties. This warning will become an error in version 0.4.11.",
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        choices = properties
        properties = {}

    properties = validate_properties(properties, expected_len=num_data)
    choices = _validate_property_choices(choices)

    # Populate the new choices by using the property keys and merging the
    # corresponding unique property and choices values.
    new_choices = {
        k: np.unique(np.concatenate((v, choices.get(k, []))))
        for k, v in properties.items()
    }

    # If there are no properties, and thus no new choices, populate new choices
    # from the input choices, and initialize property values as missing or empty.
    if len(new_choices) == 0:
        new_choices = {k: np.unique(v) for k, v in choices.items()}
        if len(new_choices) > 0:
            if num_data > 0:
                properties = {
                    k: np.array([None] * num_data) for k in new_choices
                }
            else:
                properties = {
                    k: np.empty(0, v.dtype) for k, v in new_choices.items()
                }

    # For keys that are in the input choices, but not in the new choices,
    # sometimes add appropriate array values to new choices and properties.
    if save_choices:
        for k, v in choices.items():
            if k not in new_choices:
                new_choices[k] = np.unique(v)
                properties[k] = np.array([None] * num_data)

    return properties, new_choices


def get_current_properties(
    properties: Dict[str, np.ndarray],
    choices: Dict[str, np.ndarray],
    num_data: int = 0,
) -> Dict[str, Any]:
    """Get the current property values from the properties or choices.

    Parameters
    ----------
    properties : dict[str, np.ndarray]
        The property values.
    choices : dict[str, np.ndarray]
        The property value choices.
    num_data : int
        The length of data that the properties represent (e.g. number of points).

    Returns
    -------
    dict[str, Any]
        A dictionary where the key is the property name and the value is the current
        value of that property.
    """
    current_properties = {}
    if num_data > 0:
        current_properties = {
            k: np.asarray([v[-1]]) for k, v in properties.items()
        }
    elif num_data == 0 and len(choices) > 0:
        current_properties = {
            k: np.asarray([v[0]]) for k, v in choices.items()
        }
    return current_properties


def _coerce_current_properties_value(
    value: Union[float, str, int, bool, list, tuple, np.ndarray]
) -> np.ndarray:
    """Coerce a value in a current_properties dictionary into the correct type.

    Parameters
    ----------
    value : Union[float, str, int, bool, list, tuple, np.ndarray]
        The value to be coerced.

    Returns
    -------
    coerced_value : np.ndarray
        The value in a 1D numpy array with length 1.
    """
    if isinstance(value, (np.ndarray, list, tuple)):
        if len(value) != 1:
            raise ValueError('current_properties values should have length 1.')
        coerced_value = np.asarray(value)
    else:
        coerced_value = np.array([value])

    return coerced_value


def coerce_current_properties(
    current_properties: Dict[
        str, Union[float, str, int, bool, list, tuple, np.ndarray]
    ]
) -> Dict[str, np.ndarray]:
    """Coerce a current_properties dictionary into the correct type.


    Parameters
    ----------
    current_properties : Dict[str, Union[float, str, int, bool, list, tuple, np.ndarray]]
        The current_properties dictionary to be coerced.

    Returns
    -------
    coerced_current_properties : Dict[str, np.ndarray]
        The current_properties dictionary with string keys and 1D numpy array with length 1 values.
    """
    coerced_current_properties = {
        k: _coerce_current_properties_value(v)
        for k, v in current_properties.items()
    }

    return coerced_current_properties


def dataframe_to_properties(
    dataframe: DataFrame,
) -> Dict[str, np.ndarray]:
    """Convert a dataframe to a properties dictionary.

    Parameters
    ----------
    dataframe : DataFrame
        The dataframe object to be converted to a properties dictionary

    Returns
    -------
    dict[str, np.ndarray]
        A properties dictionary where the key is the property name and the value
        is an ndarray with the property value for each point.
    """
    return {col: np.asarray(dataframe[col]) for col in dataframe}


def validate_properties(
    properties: Optional[Union[Dict[str, Array], DataFrame]],
    expected_len: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Validate the type and size of properties and coerce values to numpy arrays.

    Parameters
    ----------
    properties : dict[str, Array] or DataFrame
        The property values.
    expected_len : int
        The expected length of each property value array.

    Returns
    -------
    Dict[str, np.ndarray]
        The property values.
    """
    if properties is None or len(properties) == 0:
        return {}

    if not isinstance(properties, dict):
        properties = dataframe_to_properties(properties)

    lens = [len(v) for v in properties.values()]
    if expected_len is None:
        expected_len = lens[0]
    if any(v != expected_len for v in lens):
        raise ValueError(
            trans._(
                "the number of items must be equal for all properties",
                deferred=True,
            )
        )

    return {k: np.asarray(v) for k, v in properties.items()}


def _validate_property_choices(property_choices):
    if property_choices is None:
        property_choices = {}
    return {k: np.unique(v) for k, v in property_choices.items()}


def compute_multiscale_level(
    requested_shape, shape_threshold, downsample_factors
):
    """Computed desired level of the multiscale given requested field of view.

    The level of the multiscale should be the lowest resolution such that
    the requested shape is above the shape threshold. By passing a shape
    threshold corresponding to the shape of the canvas on the screen this
    ensures that we have at least one data pixel per screen pixel, but no
    more than we need.

    Parameters
    ----------
    requested_shape : tuple
        Requested shape of field of view in data coordinates
    shape_threshold : tuple
        Maximum size of a displayed tile in pixels.
    downsample_factors : list of tuple
        Downsampling factors for each level of the multiscale. Must be increasing
        for each level of the multiscale.

    Returns
    -------
    level : int
        Level of the multiscale to be viewing.
    """
    # Scale shape by downsample factors
    scaled_shape = requested_shape / downsample_factors

    # Find the highest resolution level allowed
    locations = np.argwhere(np.all(scaled_shape > shape_threshold, axis=1))
    if len(locations) > 0:
        level = locations[-1][0]
    else:
        level = 0
    return level


def compute_multiscale_level_and_corners(
    corner_pixels, shape_threshold, downsample_factors
):
    """Computed desired level and corners of a multiscale view.

    The level of the multiscale should be the lowest resolution such that
    the requested shape is above the shape threshold. By passing a shape
    threshold corresponding to the shape of the canvas on the screen this
    ensures that we have at least one data pixel per screen pixel, but no
    more than we need.

    Parameters
    ----------
    corner_pixels : array (2, D)
        Requested corner pixels at full resolution.
    shape_threshold : tuple
        Maximum size of a displayed tile in pixels.
    downsample_factors : list of tuple
        Downsampling factors for each level of the multiscale. Must be increasing
        for each level of the multiscale.

    Returns
    -------
    level : int
        Level of the multiscale to be viewing.
    corners : array (2, D)
        Needed corner pixels at target resolution.
    """
    requested_shape = corner_pixels[1] - corner_pixels[0]
    level = compute_multiscale_level(
        requested_shape, shape_threshold, downsample_factors
    )

    corners = corner_pixels / downsample_factors[level]
    corners = np.array([np.floor(corners[0]), np.ceil(corners[1])]).astype(int)

    return level, corners


def coerce_affine(affine, *, ndim, name=None):
    """Coerce a user input into an affine transform object.

    If the input is already an affine transform object, that same object is returned
    with a name change if the given name is not None. If the input is None, an identity
    affine transform object of the given dimensionality is returned.

    Parameters
    ----------
    affine : array-like or napari.utils.transforms.Affine
        An existing affine transform object or an array-like that is its transform matrix.
    ndim : int
        The desired dimensionality of the transform. Ignored is affine is an Affine transform object.
    name : str
        The desired name of the transform.

    Returns
    -------
    napari.utils.transforms.Affine
        The input coerced into an affine transform object.
    """
    if affine is None:
        affine = Affine(affine_matrix=np.eye(ndim + 1), ndim=ndim)
    elif isinstance(affine, np.ndarray):
        affine = Affine(affine_matrix=affine, ndim=ndim)
    elif isinstance(affine, list):
        affine = Affine(affine_matrix=np.array(affine), ndim=ndim)
    elif not isinstance(affine, Affine):
        raise TypeError(
            trans._(
                'affine input not recognized. must be either napari.utils.transforms.Affine or ndarray. Got {dtype}',
                deferred=True,
                dtype=type(affine),
            )
        )
    if name is not None:
        affine.name = name
    return affine


def dims_displayed_world_to_layer(
    dims_displayed_world: List[int],
    ndim_world: int,
    ndim_layer: int,
) -> List[int]:
    """Convert the dims_displayed from world dims to the layer dims.

    This accounts differences in the number of dimensions in the world
    dims versus the layer and for transpose and rolls.

    Parameters
    ----------
    dims_displayed_world : List[int]
        The dims_displayed in world coordinates (i.e., from viewer.dims.displayed).
    ndim_world : int
        The number of dimensions in the world coordinates (i.e., viewer.dims.ndim)
    ndim_layer : int
        The number of dimensions in layer the layer (i.e., layer.ndim).
    """
    if ndim_world > len(dims_displayed_world):
        all_dims = list(range(ndim_world))
        not_in_dims_displayed = [
            d for d in all_dims if d not in dims_displayed_world
        ]
        order = not_in_dims_displayed + dims_displayed_world
    else:
        order = dims_displayed_world
    offset = ndim_world - ndim_layer
    order = np.array(order)
    if offset <= 0:
        order = list(range(-offset)) + list(order - offset)
    else:
        order = list(order[order >= offset] - offset)
    n_display_world = len(dims_displayed_world)
    if n_display_world > ndim_layer:
        n_display_layer = ndim_layer
    else:
        n_display_layer = n_display_world
    dims_displayed = order[-n_display_layer:]

    return dims_displayed
