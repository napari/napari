from __future__ import annotations

import functools
import inspect
import sys
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import dask
import numpy as np
import pandas as pd

from napari.utils._dtype import normalize_dtype
from napari.utils.action_manager import action_manager
from napari.utils.events.custom_types import Array
from napari.utils.transforms import Affine
from napari.utils.translations import trans

if TYPE_CHECKING:
    from typing import Mapping

    import numpy.typing as npt

    from napari.layers._multiscale_data import MultiScaleData

pixel_threshold = 1e7


class Extent(NamedTuple):
    """Extent of coordinates in a local data space and world space.

    Each extent is a (2, D) array that stores the minimum and maximum coordinate
    values in each of D dimensions. Both the minimum and maximum coordinates are
    inclusive so form an axis-aligned, closed interval or a D-dimensional box
    around all the coordinates.

    Attributes
    ----------
    data : (2, D) array of floats
        The minimum and maximum raw data coordinates ignoring any transforms like
        translation or scale.
    world : (2, D) array of floats
        The minimum and maximum world coordinates after applying a transform to the
        raw data coordinates that brings them into a potentially shared world space.
    step : (D,) array of floats
        The step in each dimension that when taken from the minimum world coordinate,
        should form a regular grid that eventually hits the maximum world coordinate.
    """

    data: np.ndarray
    world: np.ndarray
    step: np.ndarray


def register_layer_action(
    keymapprovider,
    description: str,
    repeatable: bool = False,
    shortcuts: Optional[str] = None,
):
    """
    Convenient decorator to register an action with the current Layers

    It will use the function name as the action name. We force the description
    to be given instead of function docstring for translation purpose.


    Parameters
    ----------
    keymapprovider : KeymapProvider
        class on which to register the keybindings - this will typically be
        the instance in focus that will handle the keyboard shortcut.
    description : str
        The description of the action, this will typically be translated and
        will be what will be used in tooltips.
    repeatable : bool
        A flag indicating whether the action autorepeats when key is held
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
            repeatable=repeatable,
        )
        if shortcuts:
            if isinstance(shortcuts, str):
                shortcuts = [shortcuts]

            for shortcut in shortcuts:
                action_manager.bind_shortcut(name, shortcut)
        return func

    return _inner


def register_layer_attr_action(
    keymapprovider,
    description: str,
    attribute_name: str,
    shortcuts=None,
):
    """
    Convenient decorator to register an action with the current Layers.
    This will get and restore attribute from function first argument.

    It will use the function name as the action name. We force the description
    to be given instead of function docstring for translation purpose.

    Parameters
    ----------
    keymapprovider : KeymapProvider
        class on which to register the keybindings - this will typically be
        the instance in focus that will handle the keyboard shortcut.
    description : str
        The description of the action, this will typically be translated and
        will be what will be used in tooltips.
    attribute_name : str
        The name of the attribute to be restored if key is hold over `get_settings().get_settings().application.hold_button_delay.
    shortcuts : str | List[str]
        Shortcut to bind by default to the action we are registering.

    Returns
    -------
    function:
        Actual decorator to apply to a function. Given decorator returns the
        function unmodified to allow decorator stacking.

    """

    def _handle(func):
        sig = inspect.signature(func)
        try:
            first_variable_name = next(iter(sig.parameters))
        except StopIteration as e:
            raise RuntimeError(
                trans._(
                    "If actions has no arguments there is no way to know what to set the attribute to.",
                    deferred=True,
                ),
            ) from e

        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            obj = args[0] if args else kwargs[first_variable_name]
            prev_mode = getattr(obj, attribute_name)
            func(*args, **kwargs)

            def _callback():
                setattr(obj, attribute_name, prev_mode)

            return _callback

        repeatable = False  # attribute actions are always non-repeatable
        register_layer_action(
            keymapprovider, description, repeatable, shortcuts
        )(_wrapper)
        return func

    return _handle


def _nanmin(array):
    """
    call np.min but fall back to avoid nan and inf if necessary
    """
    min_value = np.min(array)
    if not np.isfinite(min_value):
        masked = array[np.isfinite(array)]
        if masked.size == 0:
            return 0
        min_value = np.min(masked)
    return min_value


def _nanmax(array):
    """
    call np.max but fall back to avoid nan and inf if necessary
    """
    max_value = np.max(array)
    if not np.isfinite(max_value):
        masked = array[np.isfinite(array)]
        if masked.size == 0:
            return 1
        max_value = np.max(masked)
    return max_value


def calc_data_range(data, rgb: bool = False) -> None | Tuple[float, float]:
    """Calculate range of data values. If all values are equal return [0, 1].

    Parameters
    ----------
    data : array
        Data to calculate range of values over.
    rgb : bool
        Flag if data is rgb.

    Returns
    -------
    values : pair of floats
        Minimum and maximum values in that order.

    Notes
    -----
    If the data type is uint8, no calculation is performed, and 0-255 is
    returned.
    """
    shape = data.shape
    chunk_size = (
        _get_chunk_size(data) if not isinstance(data, np.ndarray) else None
    )

    dtype = normalize_dtype(getattr(data, 'dtype', None))
    if dtype == np.uint8:
        return 0, 255
    if not np.issubdtype(dtype, np.integer) and chunk_size:
        return None

    if chunk_size:
        shape = _get_blocks_grid_shape(data.shape, chunk_size)

    if data.size > pixel_threshold and (
        data.ndim == 1 or (rgb and data.ndim == 2)
    ):
        reduced_data = _calc_1d_data_range(data, shape, chunk_size, rgb)
        if not reduced_data:
            return None
        if chunk_size:
            reduced_data = dask.compute(*reduced_data)
    elif data.size > pixel_threshold:
        # If data is very large take the top, bottom, and middle slices
        offset = 2 + int(rgb)
        # Indices are either numpy array or chunk indices dependent on data structure of data and are 0 < length <= 3
        idxs = _get_plane_indices(shape, offset)

        if chunk_size:
            slices = _get_crop_slices(shape, idxs, offset, chunk_size)
            if slices:
                reduced_data = [
                    [_nanmax(data[sl]) for sl in slices],
                    [_nanmin(data[sl]) for sl in slices],
                ]
                reduced_data = dask.compute(*reduced_data)
            # This is to ensure there are clim values for determining the iso threshold.
            else:
                # this is none when the chunk size product goes over the pixel threshold of 1e7
                return None
        # If we get here we are dealing with a numpy array and shape corresponds to the shape of this array.
        else:
            slices = _get_crop_slices(shape, idxs, offset)
            reduced_data = [
                [_nanmax(data[slicer[0], slicer[1]]) for slicer in slices],
                [_nanmin(data[slicer[0], slicer[1]]) for slicer in slices],
            ]
    else:
        reduced_data = data

    min_val = _nanmin(reduced_data)
    max_val = _nanmax(reduced_data)

    if min_val == max_val:
        min_val = 0
        max_val = 1
    return float(min_val), float(max_val)


def _calc_1d_data_range(data, shape, chunk_size, rgb):
    # If data is very large take the average of start, middle and end.
    n_slices = 3
    center = shape[0] // 2 * chunk_size[0] if chunk_size else shape[0] // 2
    slice_size = pixel_threshold // n_slices
    slices = [
        slice(0, slice_size),
        slice(center - int(slice_size // 2), center + int(slice_size // 2)),
        slice(-slice_size, None),
    ]
    if chunk_size:
        chunk_size_product = np.prod(chunk_size)
        allowed_chunks = int(pixel_threshold // chunk_size_product)
        multiplier = allowed_chunks // n_slices

        if rgb:
            # To ensure the chunks include all rgb channels
            chunk_size_product = chunk_size_product * shape[-1]
            allowed_chunks = pixel_threshold // chunk_size_product
            # if shape[0] >= 3:
            multiplier = allowed_chunks // n_slices
        if chunk_size_product > pixel_threshold:
            return None
        if shape[0] >= 3:
            if multiplier >= 1:
                slices = [
                    slice(0, chunk_size[0] * multiplier),
                    slice(
                        center - chunk_size[0] * (multiplier // 2 - 1),
                        center + chunk_size[0] * multiplier // 2,
                    ),
                    slice(-chunk_size[0] * multiplier, -1),
                ]
            else:
                # Means we have less than 3 allowed chunks so we just take a center slice.
                slices = [slice(center, center + chunk_size[0])]
        # due to earlier check we now data is above pixel threshold so 2 chunks is above as well
        elif shape[0] == 2:
            slices = [slice(0, chunk_size[0])]

    return [
        [_nanmax(data[sl]) for sl in slices],
        [_nanmin(data[sl]) for sl in slices],
    ]


def _get_blocks_grid_shape(
    data_shape: Sequence[int], chunk_size: Sequence[int]
) -> tuple[int]:
    """
    Get the approximate shape of the grid of array chunks.

    Gets the shape of the array not on the pixel level but on the chunk level. For example if x is 100 and
    chunk size is 10, 10 chunks fit in 100 and will thus be returned. In case of x being 105, still 10 will be
    returned. This to ensure that only whole chunks will be loaded.

    Parameters
    ----------
    data_shape: tuple[int]
        The shape of an array of chunked data.
    chunk_size: tuple[int]
        The size per dimension of the chunks in the chunked data array.

    Returns
    -------
    tuple[int]
        shape indicating the number of chunks per dimension.
    """
    return tuple(
        data_shape[i] // chunk_size[i] for i in range(len(data_shape))
    )


def _get_plane_indices(shape: Sequence[int], offset: int) -> list[tuple[int]]:
    """
    Get the indices that correspond to the lowest, middle and highest index of the non-visible dimensions in shape.

    In case the different planes are the same, duplicates are removed.

    Parameters
    ----------
    shape: Iterable[int, ...]
        Either the shape of the raw data (in pixels) or the block shape of the data (chunks)
    offset: int
        Number of visible dimensions

    Returns
    -------
    idxs: list[tuple[int], ...]
        Bottom, middle and top plane for each non-visible dimension or single plane
    """
    bottom_plane_idx = (0,) * (len(shape) - offset)
    middle_plane_idx = tuple(s // 2 for s in shape[:-offset])
    top_plane_idx = tuple(s - 1 for s in shape[:-offset])
    idxs = [bottom_plane_idx, middle_plane_idx, top_plane_idx]
    if len(idxs) != len(set(idxs)):
        return list(set(idxs))
    return idxs


def _get_crop_slices(
    shape: Sequence[int],
    plane_indices: Sequence[Sequence[int]],
    offset: int,
    chunk_shape: Optional[tuple[int]] = None,
) -> (
    None
    | list[tuple[slice, slice]]
    | list[tuple[tuple[int | slice], tuple[int | slice]]]
):
    """
    Get the crop slices to be used for determining contrast limits when data is larger than the pixel threshold.

    Parameters
    ----------
    shape: tuple[int, ...]
        Either the shape of the raw data (in pixels) or the block shape of the data (chunks)
    plane_indices: list[tuple[int, ...], ...]
        Bottom, middle and top plane or single plane index for each non-visible dimension.
    offset: int
        Number of visible dimensions.
    chunk_shape: tuple[int, ...]
        The size per dimension of the chunks.

    Returns
    -------
    slices: None | list[tuple[int | slice], ...]
        A list of crop slices.
    """
    plane_shape = shape[-offset:]

    # defaults in case we are dealing with numpy array. If not these will be overwritten.
    max_chunks_per_plane = 9
    max_allowed_chunks = len(plane_indices) * max_chunks_per_plane
    chunk_size_y = chunk_size_x = int(
        (pixel_threshold // max_allowed_chunks) ** 0.5
    )

    if chunk_shape:
        chunk_size_product = np.prod(chunk_shape)
        max_allowed_chunks = pixel_threshold // chunk_size_product
        # in case of the chunk size going over the pixel threshold, we can wait until data is in memory.
        if max_allowed_chunks == 0:
            return None
        plane_size = chunk_shape[:-offset]
        # Go from chunk indices to pixel indices.
        plane_indices = [
            (plane_index[size_index] * size,)
            for size_index, size in enumerate(plane_size)
            for plane_index in plane_indices
        ]
        chunk_plane_size = chunk_shape[-offset:]
        chunk_size_y, chunk_size_x = chunk_plane_size[0], chunk_plane_size[1]
        if len(plane_indices) != 0:
            max_chunks_per_plane = max_allowed_chunks // len(plane_indices)

        y_start_indices = _get_start_indices(plane_shape[0], chunk_size_y)
        x_start_indices = _get_start_indices(plane_shape[1], chunk_size_x)
    else:
        y_start_indices = _get_start_indices(plane_shape[0])
        x_start_indices = _get_start_indices(plane_shape[1])

    start_indices = [
        (y_start, x_start)
        for y_start in y_start_indices
        for x_start in x_start_indices
    ]
    num_start_indices = len(start_indices)
    chunk_multiplier = max(max_chunks_per_plane // num_start_indices - 1, 0)

    # We have at least 1 chunk per plane, but not all chunks are allowed to be loaded. Only lazy data
    if num_start_indices > max_chunks_per_plane >= 1:
        if num_start_indices == 3:
            # center chunk
            indices = (1,)
        # Below num_start_indices can only be 9
        else:
            if max_chunks_per_plane // 5 == 1:
                # + pattern of chunks
                indices = (1, 3, 4, 5, 7)
            elif max_chunks_per_plane // 3 == 1:
                # horizontal dashed line of chunks
                indices = (3, 4, 5)
            else:
                # center chunk
                indices = (4,)
        start_indices = [start_indices[i] for i in indices]
    elif max_chunks_per_plane < 1:
        index = 1 if num_start_indices == 3 else 5
        start_indices = [start_indices[index]]
        # get the middle plane if 3 planes exist, otherwise first plane.
        plane_indices = plane_indices[max(1, len(plane_indices) - 1)]

    slices_y = _get_slices(
        start_indices, len(y_start_indices), chunk_size_y, 0, chunk_multiplier
    )
    slices_x = _get_slices(
        start_indices, len(x_start_indices), chunk_size_x, 1, chunk_multiplier
    )
    plane_slices: list[tuple[slice, slice]] = list(zip(slices_y, slices_x))
    if len(plane_indices) == 0 or len(plane_indices[0]) == 0:
        return plane_slices
    return [
        (plane + (plane_slice[0],) + (plane_slice[1],))
        for plane in plane_indices
        for plane_slice in plane_slices
    ]


def _get_slices(
    start_indices: Iterable[Sequence[int]],
    dim_indices_length: int,
    chunk_dim_size: int,
    dim_index: int,
    multiplier: int,
) -> list[slice]:
    """
    Get the crop slices for a given dimension.

    Parameters
    ----------
    start_indices: Iterable[Sequence[int]]
        Iterable in which each element contains the start indices of given dimensions.
    dim_indices_length: int
        The original length of the indices of one specific dimension.
    chunk_dim_size: int
        Size of the slice in a particular dimension. In case of lazy data it corresponds to the size of the dimension
        of the chunks.
    dim_index: int
        The index of the dimension in the sequences in start indices for which to create the slices.
    multiplier: int
        In case of lazy data how many additional chunks to obtain.

    Returns
    -------
    list[slice]
        List of crop slices for a given dimension.
    """
    if dim_indices_length == 1:
        return [
            slice(start[dim_index], start[dim_index] + chunk_dim_size)
            for start in start_indices
        ]

    return [
        slice(
            start[dim_index] - chunk_dim_size * multiplier,
            start[dim_index] + chunk_dim_size * (multiplier + 1),
        )
        for start in start_indices
    ]


def _get_start_indices(
    dim_size: int, chunk_dim_size: Optional[int] = None
) -> list[int]:
    """
    Get the crop start indices for extracting crops in a 3 x 3 pattern.

    Gets the start indices required to extract crops in 3 x 3 pattern when appropriate. If this is not the
    case, only returns one start index.

    Parameters
    ----------
    dim_size: int
        The number of chunks or pixels of a given dimension.
    chunk_dim_size: None | int
        The size of the individual chunks for one given dimension in case data is lazy.

    Returns
    -------
    indices: list[int, ...]
        The chunk start indices in array indices along a given dimension.
    """
    if dim_size <= 2:
        indices = [0]
    elif dim_size == 3:
        indices = (
            [0, chunk_dim_size, chunk_dim_size * 2]
            if chunk_dim_size
            else [0, 1, 2]
        )
    else:
        quarter_index = (
            dim_size // 4 * chunk_dim_size if chunk_dim_size else dim_size // 4
        )
        center_index = quarter_index * 2
        qthree_index = quarter_index * 3
        indices = [quarter_index, center_index, qthree_index]
    return indices


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
        normal = np.array([d[1], -d[0]]) if len(d) == 2 else np.cross(d, p)
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


def convert_to_uint8(data: np.ndarray) -> Optional[np.ndarray]:
    """
    Convert array content to uint8, always returning a copy.

    Based on skimage.util.dtype._convert but limited to an output type uint8,
    so should be equivalent to skimage.util.dtype.img_as_ubyte.

    If all negative, values are clipped to 0.

    If values are integers and below 256, this simply casts.
    Otherwise the maximum value for the input data type is determined and
    output values are proportionally scaled by this value.

    Binary images are converted so that False -> 0, True -> 255.

    Float images are multiplied by 255 and then cast to uint8.
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
        image_out = np.nan_to_num(image_out, copy=False)
        return image_out.astype(out_dtype)

    if in_kind in "ui":
        if in_kind == "u":
            if data.max() < out_max:
                return data.astype(out_dtype)
            return np.right_shift(data, (data.dtype.itemsize - 1) * 8).astype(
                out_dtype
            )

        np.maximum(data, 0, out=data, dtype=data.dtype)
        if data.dtype == np.int8:
            return (data * 2).astype(np.uint8)
        if data.max() < out_max:
            return data.astype(out_dtype)
        return np.right_shift(data, (data.dtype.itemsize - 1) * 8 - 1).astype(
            out_dtype
        )
    return None


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
    current_properties: Mapping[
        str, Union[float, str, int, bool, list, tuple, npt.NDArray]
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

    # Find the highest level (lowest resolution) allowed
    locations = np.argwhere(np.all(scaled_shape > shape_threshold, axis=1))
    level = locations[-1][0] if len(locations) > 0 else 0
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

    order_arr = np.array(order)
    if offset <= 0:
        order = list(range(-offset)) + list(order_arr - offset)
    else:
        order = list(order_arr[order_arr >= offset] - offset)

    n_display_world = len(dims_displayed_world)
    if n_display_world > ndim_layer:
        n_display_layer = ndim_layer
    else:
        n_display_layer = n_display_world
    dims_displayed = order[-n_display_layer:]

    return dims_displayed


def get_extent_world(data_extent, data_to_world, centered=None):
    """Range of layer in world coordinates base on provided data_extent

    Parameters
    ----------
    data_extent : array, shape (2, D)
        Extent of layer in data coordinates.
    data_to_world : napari.utils.transforms.Affine
        The transform from data to world coordinates.

    Returns
    -------
    extent_world : array, shape (2, D)
    """
    if centered is not None:
        warnings.warn(
            trans._(
                'The `centered` argument is deprecated. '
                'Extents are now always centered on data points.',
                deferred=True,
            ),
            stacklevel=2,
        )

    D = data_extent.shape[1]
    full_data_extent = np.array(np.meshgrid(*data_extent.T)).T.reshape(-1, D)
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


class _FeatureTable:
    """Stores feature values and their defaults.

    Parameters
    ----------
    values : Optional[Union[Dict[str, np.ndarray], pd.DataFrame]]
        The features values, which will be passed to the pandas DataFrame initializer.
        If this is a pandas DataFrame with a non-default index, that index
        (except its length) will be ignored.
    num_data : Optional[int]
        The number of the elements in the layer calling this, such as
        the number of points, which is used to check that the features
        table has the expected number of rows. If None, then the default
        DataFrame index is used.
    defaults: Optional[Union[Dict[str, Any], pd.DataFrame]]
        The default feature values, which if specified should have the same keys
        as the values provided. If None, will be inferred from the values.
    """

    def __init__(
        self,
        values: Optional[Union[Dict[str, np.ndarray], pd.DataFrame]] = None,
        *,
        num_data: Optional[int] = None,
        defaults: Optional[Union[Dict[str, Any], pd.DataFrame]] = None,
    ) -> None:
        self._values = _validate_features(values, num_data=num_data)
        self._defaults = _validate_feature_defaults(defaults, self._values)

    @property
    def values(self) -> pd.DataFrame:
        """The feature values table."""
        return self._values

    def set_values(self, values, *, num_data=None) -> None:
        """Sets the feature values table."""
        self._values = _validate_features(values, num_data=num_data)
        self._defaults = _validate_feature_defaults(None, self._values)

    @property
    def defaults(self) -> pd.DataFrame:
        """The default values one-row table."""
        return self._defaults

    def set_defaults(
        self, defaults: Union[Dict[str, Any], pd.DataFrame]
    ) -> None:
        """Sets the feature default values."""
        self._defaults = _validate_feature_defaults(defaults, self._values)

    def properties(self) -> Dict[str, np.ndarray]:
        """Converts this to a deprecated properties dictionary.

        This will reference the features data when possible, but in general the
        returned dictionary may contain copies of those data.

        Returns
        -------
        Dict[str, np.ndarray]
            The properties dictionary equivalent to the given features.
        """
        return _features_to_properties(self._values)

    def choices(self) -> Dict[str, np.ndarray]:
        """Converts this to a deprecated property choices dictionary.

        Only categorical features will have corresponding entries in the dictionary.

        Returns
        -------
        Dict[str, np.ndarray]
            The property choices dictionary equivalent to this.
        """
        return {
            name: series.dtype.categories.to_numpy()
            for name, series in self._values.items()
            if isinstance(series.dtype, pd.CategoricalDtype)
        }

    def currents(self) -> Dict[str, np.ndarray]:
        """Converts the defaults table to a deprecated current properties dictionary."""
        return _features_to_properties(self._defaults)

    def set_currents(
        self,
        currents: Dict[str, npt.NDArray],
        *,
        update_indices: Optional[List[int]] = None,
    ) -> None:
        """Sets the default values using the deprecated current properties dictionary.

        May also update some of the feature values to be equal to the new default values.

        Parameters
        ----------
        currents : Dict[str, np.ndarray]
            The new current property values.
        update_indices : Optional[List[int]]
            If not None, the all features values at the given row indices will be set to
            the corresponding new current/default feature values.
        """
        currents = coerce_current_properties(currents)
        self._defaults = _validate_features(currents, num_data=1)
        if update_indices is not None:
            for k in self._defaults:
                self._values[k][update_indices] = self._defaults[k][0]

    def resize(
        self,
        size: int,
    ) -> None:
        """Resize this padding with default values if required.

        Parameters
        ----------
        size : int
            The new size (number of rows) of the features table.
        """
        current_size = self._values.shape[0]
        if size < current_size:
            self.remove(range(size, current_size))
        elif size > current_size:
            to_append = self._defaults.iloc[np.zeros(size - current_size)]
            self.append(to_append)

    def append(self, to_append: pd.DataFrame) -> None:
        """Append new feature rows to this.

        Parameters
        ----------
        to_append : pd.DataFrame
            The features to append.
        """
        self._values = pd.concat([self._values, to_append], ignore_index=True)

    def remove(self, indices: Any) -> None:
        """Remove rows from this by index.

        Parameters
        ----------
        indices : Any
            The indices of the rows to remove. Must be usable as the labels parameter
            to pandas.DataFrame.drop.
        """
        self._values = self._values.drop(labels=indices, axis=0).reset_index(
            drop=True
        )

    def reorder(self, order: Sequence[int]) -> None:
        """Reorders the rows of the feature values table."""
        self._values = self._values.iloc[order].reset_index(drop=True)

    @classmethod
    def from_layer(
        cls,
        *,
        features: Optional[Union[Dict[str, np.ndarray], pd.DataFrame]] = None,
        feature_defaults: Optional[Union[Dict[str, Any], pd.DataFrame]] = None,
        properties: Optional[
            Union[Dict[str, np.ndarray], pd.DataFrame]
        ] = None,
        property_choices: Optional[Dict[str, np.ndarray]] = None,
        num_data: Optional[int] = None,
    ) -> _FeatureTable:
        """Coerces a layer's keyword arguments to a feature manager.

        Parameters
        ----------
        features : Optional[Union[Dict[str, np.ndarray], pd.DataFrame]]
            The features input to a layer.
        properties : Optional[Union[Dict[str, np.ndarray], pd.DataFrame]]
            The properties input to a layer.
        property_choices : Optional[Dict[str, np.ndarray]]
            The property choices input to a layer.
        num_data : Optional[int]
            The number of the elements in the layer calling this, such as
            the number of points.

        Returns
        -------
        _FeatureTable
            The feature manager created from the given layer keyword arguments.

        Raises
        ------
        ValueError
            If the input property columns are not all the same length, or if
            that length is not equal to the given num_data.
        """
        if properties is not None or property_choices is not None:
            features = _features_from_properties(
                properties=properties,
                property_choices=property_choices,
                num_data=num_data,
            )
        return cls(features, defaults=feature_defaults, num_data=num_data)


def _get_default_column(column: pd.Series) -> pd.Series:
    """Get the default column of length 1 from a data column."""
    value = None
    if column.size > 0:
        value = column.iloc[-1]
    elif isinstance(column.dtype, pd.CategoricalDtype):
        choices = column.dtype.categories
        if choices.size > 0:
            value = choices[0]
    elif isinstance(column.dtype, np.dtype) and np.issubdtype(
        column.dtype, np.integer
    ):
        # For numpy backed columns that store integers there's no way to
        # store missing values, so passing None creates an np.float64 series
        # containing NaN. Therefore, use a default of 0 instead.
        value = 0
    return pd.Series(data=value, dtype=column.dtype, index=range(1))


def _validate_features(
    features: Optional[Union[Dict[str, np.ndarray], pd.DataFrame]],
    *,
    num_data: Optional[int] = None,
) -> pd.DataFrame:
    """Validates and coerces feature values into a pandas DataFrame.

    See Also
    --------
    :class:`_FeatureTable` : See initialization for parameter descriptions.
    """
    if isinstance(features, pd.DataFrame):
        features = features.reset_index(drop=True)
    elif isinstance(features, dict):
        # Convert all array-like objects into a numpy array.
        # This section was introduced due to an unexpected behavior when using
        # a pandas Series with mixed indices as input.
        # This way should handle all array-like objects correctly.
        # See https://github.com/napari/napari/pull/4755 for more details.
        features = {
            key: np.array(value, copy=False) for key, value in features.items()
        }
    index = None if num_data is None else range(num_data)
    return pd.DataFrame(data=features, index=index)


def _validate_feature_defaults(
    defaults: Optional[Union[Dict[str, Any], pd.DataFrame]],
    values: pd.DataFrame,
) -> pd.DataFrame:
    """Validates and coerces feature default values into a pandas DataFrame.

    See Also
    --------
    :class:`_FeatureTable` : See initialization for parameter descriptions.
    """
    if defaults is None:
        defaults = {c: _get_default_column(values[c]) for c in values.columns}
    else:
        default_columns = set(defaults.keys())
        value_columns = set(values.keys())
        extra_defaults = default_columns - value_columns
        if len(extra_defaults) > 0:
            raise ValueError(
                trans._(
                    'Feature defaults contain some extra columns not in feature values: {extra_defaults}',
                    deferred=True,
                    extra_defaults=extra_defaults,
                )
            )
        missing_defaults = value_columns - default_columns
        if len(missing_defaults) > 0:
            raise ValueError(
                trans._(
                    'Feature defaults is missing some columns in feature values: {missing_defaults}',
                    deferred=True,
                    missing_defaults=missing_defaults,
                )
            )
        # Convert to series first to capture the per-column dtype from values,
        # since the DataFrame initializer does not support passing multiple dtypes.
        defaults = {
            c: pd.Series(
                defaults[c],
                dtype=values.dtypes[c],
                index=range(1),
            )
            for c in defaults
        }

    return pd.DataFrame(defaults, index=range(1))


def _features_from_properties(
    *,
    properties: Optional[Union[Dict[str, np.ndarray], pd.DataFrame]] = None,
    property_choices: Optional[Dict[str, np.ndarray]] = None,
    num_data: Optional[int] = None,
) -> pd.DataFrame:
    """Validates and coerces deprecated properties input into a features DataFrame.

    See Also
    --------
    :meth:`_FeatureTable.from_layer`
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


def _features_to_properties(features: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Converts a features DataFrame to a deprecated properties dictionary.

    See Also
    --------
    :meth:`_FeatureTable.properties`
    """
    return {name: series.to_numpy() for name, series in features.items()}


def _unique_element(array: Array) -> Optional[Any]:
    """
    Returns the unique element along the 0th axis, if it exists; otherwise, returns None.

    This is faster than np.unique, does not require extra tricks for nD arrays, and
    does not fail for non-sortable elements.
    """
    if len(array) == 0:
        return None
    el = array[0]
    if np.any(array[1:] != el):
        return None
    return el


def _get_chunk_size(
    data: MultiScaleData
    | Iterable
    | npt.NDArray
    | int
    | float
    | list
    | Iterable[npt.NDArray]
    | None,
) -> None | tuple[int]:
    """Get chunk size from a given layer.

    Parameters
    ----------
    layer : napari.layers.Image
        Layer to determine chunk size for.
    Returns
    -------
    chunk_size : tuple or None
        Chunk size for the layer.
    """
    if isinstance(data, np.ndarray):
        return None

    if "zarr" in sys.modules:
        from zarr.core import Array as ZarrArray

        if isinstance(data, ZarrArray):
            return data.chunks

    if "dask" in sys.modules:
        from dask.array import Array as DaskArray

        if isinstance(data, DaskArray):
            return data.chunksize

    if "tensorstore" in sys.modules:
        from tensorstore import TensorStore

        if isinstance(data, TensorStore):
            # TensorStore allow to specify different read and write chunk sizes
            # we use the read chunk size to have same chunk size for labels like
            # when load data from drive
            return data.chunk_layout.read_chunk.shape
        return None

    if "xarray" in sys.modules:
        from xarray import DataArray

        if isinstance(data, DataArray):
            chunk_size = data.chunksizes
            if len(chunk_size) != 0:
                return tuple(dim_chunk[0] for dim_chunk in chunk_size)
    return None
