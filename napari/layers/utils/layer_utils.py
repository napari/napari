from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import dask
import numpy as np
import pandas as pd

from ...utils.action_manager import action_manager
from ...utils.events.custom_types import Array
from ...utils.transforms import Affine
from ...utils.translations import trans


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


def _nanmin(array):
    """
    call np.min but fall back to avoid nan and inf if necessary
    """
    min = np.min(array)
    if not np.isfinite(min):
        masked = array[np.isfinite(array)]
        if masked.size == 0:
            return 0
        min = np.min(masked)
    return min


def _nanmax(array):
    """
    call np.max but fall back to avoid nan and inf if necessary
    """
    max = np.max(array)
    if not np.isfinite(max):
        masked = array[np.isfinite(array)]
        if masked.size == 0:
            return 1
        max = np.max(masked)
    return max


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

    if data.size > 1e7 and (data.ndim == 1 or (rgb and data.ndim == 2)):
        # If data is very large take the average of start, middle and end.
        center = int(data.shape[0] // 2)
        slices = [
            slice(0, 4096),
            slice(center - 2048, center + 2048),
            slice(-4096, None),
        ]
        reduced_data = [
            [_nanmax(data[sl]) for sl in slices],
            [_nanmin(data[sl]) for sl in slices],
        ]
    elif data.size > 1e7:
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
            # Find a central patch of the image to take
            center = [int(s // 2) for s in data.shape[-offset:]]
            central_slice = tuple(slice(c - 31, c + 31) for c in center[:2])
            reduced_data = [
                [_nanmax(data[idx + central_slice]) for idx in idxs],
                [_nanmin(data[idx + central_slice]) for idx in idxs],
            ]
        else:
            reduced_data = [
                [_nanmax(data[idx]) for idx in idxs],
                [_nanmin(data[idx]) for idx in idxs],
            ]
        # compute everything in one go
        reduced_data = dask.compute(*reduced_data)
    else:
        reduced_data = data

    min_val = _nanmin(reduced_data)
    max_val = _nanmax(reduced_data)

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
    properties: Optional[Union[Dict[str, Array], pd.DataFrame]],
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


def dataframe_to_properties(
    dataframe: pd.DataFrame,
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
    properties: Optional[Union[Dict[str, Array], pd.DataFrame]],
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
            raise ValueError(
                trans._(
                    'current_properties values should have length 1.',
                    deferred=True,
                )
            )
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


def get_extent_world(data_extent, data_to_world, centered=False):
    """Range of layer in world coordinates base on provided data_extent

    Parameters
    ----------
    data_extent : array, shape (2, D)
        Extent of layer in data coordinates.
    data_to_world : napari.utils.transforms.Affine
        The transform from data to world coordinates.
    centered : bool
        If pixels should be centered. By default False.

    Returns
    -------
    extent_world : array, shape (2, D)
    """
    D = data_extent.shape[1]
    # subtract 0.5 to get from pixel center to pixel edge
    offset = 0.5 * bool(centered)
    pixel_extents = tuple(d - offset for d in data_extent.T)

    full_data_extent = np.array(np.meshgrid(*pixel_extents)).T.reshape(-1, D)
    full_world_extent = data_to_world(full_data_extent)
    world_extent = np.array(
        [
            np.min(full_world_extent, axis=0),
            np.max(full_world_extent, axis=0),
        ]
    )
    return world_extent


def features_to_pandas_dataframe(features: Any) -> pd.DataFrame:
    """Coerces a layer's features property to a pandas DataFrame.

    In general, this may copy the data from features into the returned
    DataFrame so there is no guarantee that changing element values in the
    returned DataFrame will also change values in the features property.

    Parameters
    ----------
    features
        The features property of a layer.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame that stores the given features.
    """
    return features


def _validate_features(
    features: Union[Dict[str, np.ndarray], pd.DataFrame],
    *,
    num_data: Optional[int] = None,
) -> pd.DataFrame:
    """Validates and coerces a features table into a pandas DataFrame.

    Parameters
    ----------
    features : Union[Dict[str, np.ndarray], pd.DataFrame]
        The features table input, which will be passed to the pandas
        DataFrame initializer. If this is a pandas DataFrame with a
        non-default index, that index (except its length) will be
        ignored.
    num_data : Optional[int]
        The number of the elements in the layer calling this, such as
        the number of points, which is used to check that the features
        table has the expected number of rows. If None, then the default
        DataFrame index is used.

    Returns
    -------
    pd.DataFrame
        The pandas DataFrame created from the input features table. This
        may or may not store a copy of the input data.

    Raises
    ------
    ValueError
        If the input feature columns are not all the same length, or if
        that length is not equal to the given num_data.
    """
    if isinstance(features, pd.DataFrame):
        features = features.reset_index(drop=True)
    index = None if num_data is None else range(num_data)
    return pd.DataFrame(data=features, index=index)


def _features_from_properties(
    *,
    properties: Optional[Union[Dict[str, np.ndarray], pd.DataFrame]] = None,
    property_choices: Optional[Dict[str, np.ndarray]] = None,
    num_data: Optional[int] = None,
) -> pd.DataFrame:
    """Validates and coerces deprecated properties input into a features DataFrame.

    Parameters
    ----------
    properties : Dict[str, np.ndarray]
        The properties of a layer.
    property_choices : Dict[str, np.ndarray]
        The property choices of a layer.
    num_data : Optional[int]
        The number of the elements in the layer calling this, such as
        the number of points.

    Returns
    -------
    pd.DataFrame
        The pandas DataFrame created from the input features table.
        If the input features are already a DataFrame, the data will not
        be copied, otherwise they will.

    Raises
    ------
    ValueError
        If the input properties columns are not all the same length, or if
        that length is not equal to the given num_data.
    """
    # Create categorical series for any choices provided.
    if property_choices is not None:
        properties = pd.DataFrame(data=properties)
        for name, choices in property_choices.items():
            dtype = pd.CategoricalDtype(categories=choices)
            num_values = properties.shape[0] if num_data is None else num_data
            values = (
                properties[name] if name in properties else [None] * num_values
            )
            properties[name] = pd.Series(values, dtype=dtype)
    return _validate_features(properties, num_data=num_data)


def _features_to_choices(features: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Converts a features DataFrame to a deprecated property choices dictionary.

    Only categorical features will have corresponding entries in the dictionary.

    Parameters
    ----------
    features : pd.DataFrame
        The features of a layer.

    Returns
    -------
    Dict[str, np.ndarray]
        The property choices dictionary equivalent to the given features.
    """
    return {
        name: series.dtype.categories.to_numpy()
        for name, series in features.items()
        if isinstance(series.dtype, pd.CategoricalDtype)
    }


def _features_to_properties(features: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Converts a features DataFrame to a deprecated properties dictionary.

    This will reference the features data when possible, but in general the
    returned dictionary mary contain copies of those data.

    Parameters
    ----------
    features : pd.DataFrame
        The features of a layer.

    Returns
    -------
    Dict[str, np.ndarray]
        The properties dictionary equivalent to the given features.
    """
    return {name: series.to_numpy() for name, series in features.items()}


def _resize_features(
    features: pd.DataFrame,
    size: int,
    *,
    current_values: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Resize a features DataFrame to a new size, padding with default values if required.

    Parameters
    ----------
    features : pd.DataFrame
        The features of a layer.
    size : int
        The new size (number of rows) of the features table.
    current_values : Dict[str, np.ndarray]
        The current or default value for each feature stored in a length-1 array.
        If the new size is greater than the current, these default
        values will be repeated for the new rows. If a feature is
        missing from this dictionary, missing values will be used instead.

    Returns
    -------
    pd.DataFrame
        The new resized features table. In general, this will be a copy, except
        when the size does not change.
    """
    current_size = features.shape[0]
    if size < current_size:
        return _remove_features(features, range(size, current_size))
    elif size > current_size:
        num_append = size - current_size
        to_append = pd.DataFrame(
            {
                name: np.repeat(current_values.get(name), num_append, axis=0)
                for name in features
            },
            index=range(num_append),
        )
        return _append_features(features, to_append)
    return features


def _append_features(
    features: pd.DataFrame, to_append: pd.DataFrame
) -> pd.DataFrame:
    """Append new feature rows to an existing features table.

    Parameters
    ----------
    features : pd.DataFrame
        The features of a layer.
    to_append : pd.DataFrame
        The features to append.

    Returns
    -------
    pd.DataFrame
        The resulting features table, which contain copies of the existing
        and given data.
    """
    return features.append(to_append, ignore_index=True)


def _remove_features(features: pd.DataFrame, indices: Any) -> pd.DataFrame:
    """Remove rows from a features table by index.

    Parameters
    ----------
    features : pd.DataFrame
        The features of a layer.
    indices : Any
        The indices of the rows to remove. Must be usable as the labels parameter
        to pandas.DataFrame.drop.

    Returns
    -------
    pd.DataFrame
        The resulting features table, which contain copies of the existing data.
    """
    return features.drop(labels=indices, axis=0).reset_index(drop=True)
