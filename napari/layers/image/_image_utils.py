import numpy as np


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

    return ndim > 2 and last_dim < 5


def guess_multiscale(data):
    """Guess if the passed data is multiscale of not.

    If shape of arrays along first axis is strictly decreasing.

    Parameters
    ----------
    data : array or list of array
        Data that should be checked.

    Returns
    -------
    bool
        If data is multiscale or not.
    """
    # If the data has ndim and is not one-dimensional then cannot be multiscale
    if hasattr(data, 'ndim') and data.ndim > 1:
        return False

    size = np.array([np.prod(d.shape, dtype=np.uint64) for d in data])
    if len(size) > 1:
        return bool(np.all(size[:-1] > size[1:]))
    else:
        return False


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
