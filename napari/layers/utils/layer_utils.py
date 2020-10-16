from typing import Dict, Tuple, Union

import dask
import numpy as np

from ...utils.colormaps import Colormap


def calc_data_range(data):
    """Calculate range of data values. If all values are equal return [0, 1].

    Parameters
    ----------
    data : array
        Data to calculate range of values over.

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
    if np.prod(data.shape) > 1e6:
        # If data is very large take the average of the top, bottom, and
        # middle slices
        bottom_plane_idx = (0,) * (data.ndim - 2)
        middle_plane_idx = tuple(s // 2 for s in data.shape[:-2])
        top_plane_idx = tuple(s - 1 for s in data.shape[:-2])
        idxs = [bottom_plane_idx, middle_plane_idx, top_plane_idx]
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


def dataframe_to_properties(dataframe) -> Dict[str, np.ndarray]:
    """Convert a dataframe to Points.properties formatted dictionary.

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

    properties = {col: np.asarray(dataframe[col]) for col in dataframe}
    index = None
    if 'index' in properties:
        index = {i: k for k, i in enumerate(properties['index'])}
    return properties, index


def guess_continuous(property: np.ndarray) -> bool:
    """Guess if the property is continuous (return True) or categorical (return False)"""
    # if the property is a floating type, guess continuous
    if (
        issubclass(property.dtype.type, np.floating)
        or len(np.unique(property)) > 16
    ):
        return True
    else:
        return False


def map_property(
    prop: np.ndarray,
    colormap: Colormap,
    contrast_limits: Union[None, Tuple[float, float]] = None,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Apply a colormap to a property

    Parameters
    ----------
    prop : np.ndarray
        The property to be colormapped
    colormap : napari.utils.Colormap
        The colormap object to apply to the property
    contrast_limits : Union[None, Tuple[float, float]]
        The contrast limits for applying the colormap to the property.
        If a 2-tuple is provided, it should be provided as (lower_bound, upper_bound).
        If None is provided, the contrast limits will be set to (property.min(), property.max()).
        Default value is None.
    """

    if contrast_limits is None:
        contrast_limits = (prop.min(), prop.max())
    normalized_properties = np.interp(prop, contrast_limits, (0, 1))
    mapped_properties = colormap.map(normalized_properties)

    return mapped_properties, contrast_limits


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
