from functools import lru_cache

import numpy as np


def interpolate_coordinates(old_coord, new_coord, brush_size):
    """Interpolates coordinates depending on brush size.

    Useful for ensuring painting is continuous in labels layer.

    Parameters
    ----------
    old_coord : np.ndarray, 1x2
        Last position of cursor.
    new_coord : np.ndarray, 1x2
        Current position of cursor.
    brush_size : float
        Size of brush, which determines spacing of interpolation.

    Returns
    -------
    coords : np.array, Nx2
        List of coordinates to ensure painting is continuous
    """
    num_step = round(
        max(abs(np.array(new_coord) - np.array(old_coord))) / brush_size * 4
    )
    coords = [
        np.linspace(old_coord[i], new_coord[i], num=int(num_step + 1))
        for i in range(len(new_coord))
    ]
    coords = np.stack(coords).T
    if len(coords) > 1:
        coords = coords[1:]

    return coords


@lru_cache(maxsize=64)
def sphere_indices(radius, sphere_dims):
    """Generate centered indices within circle or n-dim sphere.

    Parameters
    -------
    radius : float
        Radius of circle/sphere
    sphere_dims : int
        Number of circle/sphere dimensions

    Returns
    -------
    mask_indices : array
        Centered indices within circle/sphere
    """
    # Create multi-dimensional grid to check for
    # circle/membership around center
    vol_radius = radius + 0.5

    indices_slice = [slice(-vol_radius, vol_radius + 1)] * sphere_dims
    indices = np.mgrid[indices_slice].T.reshape(-1, sphere_dims)
    distances_sq = np.sum(indices ** 2, axis=1)
    # Use distances within desired radius to mask indices in grid
    mask_indices = indices[distances_sq <= radius ** 2].astype(int)

    return mask_indices


def indices_in_shape(idxs, shape):
    """Return idxs after filtering out indices that are not in given shape.

    Parameters
    ----------
    idxs : tuple of array of int, or 2D array of int
        The input coordinates. These should be in one of two formats:

        - a tuple of 1D arrays, as for NumPy fancy indexing, or
        - a 2D array of shape (ncoords, ndim), as a list of coordinates

    shape : tuple of int
        The shape in which all indices must fit.

    Returns
    -------
    idxs_filtered : tuple of array of int, or 2D array of int
        The subset of the input idxs that falls within shape.

    Examples
    --------
    >>> idxs0 = (np.array([5, 45, 2]), np.array([6, 5, -5]))
    >>> indices_in_shape(idxs0, (10, 10))
    (array([5]), array([6]))
    >>> idxs1 = np.transpose(idxs0)
    >>> indices_in_shape(idxs1, (10, 10))
    array([[5, 6]])
    """
    np_index = isinstance(idxs, tuple)
    if np_index:  # normalize to 2D coords array
        idxs = np.transpose(idxs)
    keep_coords = np.logical_and(
        np.all(idxs >= 0, axis=1), np.all(idxs < np.array(shape), axis=1)
    )
    filtered = idxs[keep_coords]
    if np_index:  # convert back to original format
        filtered = tuple(filtered.T)
    return filtered


def get_dtype(layer):
    """Returns dtype of layer data

    Parameters
    ----------
    layer : Labels
        Labels layer (may be multiscale)

    Returns
    -------
    dtype
        dtype of Layer data
    """
    layer_data = layer.data
    if not isinstance(layer_data, list):
        layer_data = [layer_data]
    layer_data_level = layer_data[0]
    if hasattr(layer_data_level, 'dtype'):
        layer_dtype = layer_data_level[0].dtype
    else:
        layer_dtype = type(layer_data_level)

    return layer_dtype
