"""guess_rgb, guess_multiscale, guess_labels.
"""
from typing import Tuple

import numpy as np

from napari.layers._data_protocols import LayerDataProtocol
from napari.layers._multiscale_data import MultiScaleData
from napari.utils.translations import trans


def guess_rgb(shape):
    """Guess if the passed shape comes from rgb data.

    If last dim is 3 or 4 assume the data is rgb, including rgba.

    Parameters
    ----------
    shape : list of int
        Shape of the data that should be checked.

    Returns
    -------
    bool
        If data is rgb or not.
    """
    ndim = len(shape)
    last_dim = shape[-1]

    return ndim > 2 and last_dim in (3, 4)


def guess_multiscale(data) -> Tuple[bool, LayerDataProtocol]:
    """Guess whether the passed data is multiscale, process it accordingly.

    If shape of arrays along first axis is strictly decreasing, the data is
    multiscale. If it is the same shape everywhere, it is not. Various
    ambiguous conditions in between will result in a ValueError being raised,
    or in an "unwrapping" of data, if data contains only one element.

    Parameters
    ----------
    data : array or list of array
        Data that should be checked.

    Returns
    -------
    multiscale : bool
        True if the data is thought to be multiscale, False otherwise.
    data : list or array
        The input data, perhaps with the leading axis removed.
    """
    # If the data has ndim and is not one-dimensional then cannot be multiscale
    # If data is a zarr array, this check ensure that subsets of it are not
    # instantiated. (`for d in data` instantiates `d` as a NumPy array if
    # `data` is a zarr array.)
    if isinstance(data, MultiScaleData):
        return True, data

    if hasattr(data, 'ndim') and data.ndim > 1:
        return False, data

    if isinstance(data, (list, tuple)) and len(data) == 1:
        # pyramid with only one level, unwrap
        return False, data[0]

    shapes = [d.shape for d in data]
    sizes = np.array([np.prod(shape, dtype='object') for shape in shapes])
    if len(sizes) <= 1:
        return False, data

    consistent = bool(np.all(sizes[:-1] > sizes[1:]))
    if np.all(sizes == sizes[0]):
        # note: the individual array case should be caught by the first
        # code line in this function, hasattr(ndim) and ndim > 1.
        raise ValueError(
            trans._(
                'Input data should be an array-like object, or a sequence of arrays of decreasing size. Got arrays of single shape: {shape}',
                deferred=True,
                shape=shapes[0],
            )
        )
    if not consistent:
        raise ValueError(
            trans._(
                'Input data should be an array-like object, or a sequence of arrays of decreasing size. Got arrays in incorrect order, shapes: {shapes}',
                deferred=True,
                shapes=shapes,
            )
        )

    return True, MultiScaleData(data)


def guess_labels(data):
    """Guess if array contains labels data."""

    if hasattr(data, 'dtype') and data.dtype in (
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    ):
        return 'labels'

    return 'image'
