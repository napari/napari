from typing import Dict

import dask
import numpy as np


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
