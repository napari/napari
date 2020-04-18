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

    if ndim > 2 and last_dim < 5:
        return True
    else:
        return False


def guess_pyramid(data):
    """Guess if the passed data is a pyramid of not.

    If shape of arrays along first axis is strictly decreasing.

    Parameters
    ----------
    data : array or list of array
        Data that should be checked.

    Returns
    -------
    bool
        If data is pyramid or not.
    """
    # If the data has ndim and is not one-dimensional then cannot be pyramid
    if hasattr(data, 'ndim') and data.ndim > 1:
        return False

    size = np.array([np.prod(d.shape, dtype=np.uint64) for d in data])
    if len(size) > 1:
        return np.all(size[:-1] > size[1:])
    else:
        return False
