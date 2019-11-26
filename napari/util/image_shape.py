import numpy as np
from scipy import ndimage as ndi


def guess_rgb(shape):
    """If last dim is 3 or 4 assume image is rgb.
    """
    ndim = len(shape)
    last_dim = shape[-1]

    if ndim > 2 and last_dim < 5:
        return True
    else:
        return False


def get_ndim_and_rgb(data_shape, rgb):
    """Determine dimensionality of data and if data is rgb.

    Parameters
    ----------
    data_shape : list
        List of data shape on each axis.
    rgb : bool, None
        Current guess if data is rgb or not specified.

    Returns
    -------
    ndim : int
        Dimensionality of the data.
    rgb : bool
        `True` if data is rgb, `False` otherwise.
    """
    # Determine if rgb, and determine dimensionality
    if rgb is False:
        pass
    else:
        # If rgb is True or None then guess if rgb
        # allowed or not, and if allowed set it to be True
        rgb_guess = guess_rgb(data_shape)
        if rgb and rgb_guess is False:
            raise ValueError(
                "Non rgb or rgba data was passed, but rgb data was"
                " requested."
            )
        else:
            rgb = rgb_guess

    if rgb:
        ndim = len(data_shape) - 1
    else:
        ndim = len(data_shape)

    return ndim, rgb


def guess_pyramid(data):
    """If shape of arrays along first axis is strictly decreasing.
    """
    # If the data has ndim and is not one-dimensional then cannot be pyramid
    if hasattr(data, 'ndim') and data.ndim > 1:
        return False

    size = np.array([np.prod(d.shape, dtype=np.uint64) for d in data])
    if len(size) > 1:
        return bool(np.all(size[:-1] > size[1:]))
    else:
        return False


def trim_pyramid(pyramid):
    """Trim very small arrays of top of pyramid.

    Parameters
    ----------
    pyramid : list of array
        Pyramid data

    Returns
    -------
    trimmed : list of array
        Trimmed pyramid data
    """
    keep = [np.any(np.greater_equal(p.shape, 2 ** 6 - 1)) for p in pyramid]
    if np.sum(keep) >= 2:
        return [p for k, p in zip(keep, pyramid) if k]
    else:
        return pyramid[:2]


def should_be_pyramid(shape):
    """Check if any data axes needs to be pyramidified

    Parameters
    ----------
    shape : tuple of int
        Shape of data to be tested

    Returns
    -------
    pyr_axes : tuple of bool
        True wherever an axis exceeds the pyramid threshold.
    """
    return np.log2(shape) >= 13


def make_pyramid(data, pyr_axes):
    """Check if data is or needs to be a pyramid and make one if needed.

    Parameters
    ----------
    data : array
        Data to be turned into pyramid.
    pyr_axes : list
        Axes to do pyramiding on.

    Returns
    -------
    data_pyramid : list
        List of arrays where each array is a level of the pyramid.
    """
    # Set axes to be downsampled to have a factor of 2
    downscale = np.ones(len(data.shape))
    downscale[pyr_axes] = 2
    largest = np.min(np.array(data.shape)[pyr_axes])
    # Determine number of downsample steps needed
    max_layer = np.floor(np.log2(largest) - 9).astype(int)
    data_pyramid = fast_pyramid(data, downscale=downscale, max_layer=max_layer)
    return data_pyramid


def get_pyramid(data, is_pyramid=None, force_pyramid=False):
    """Check if data is or needs to be a pyramid and make one if needed.

    Parameters
    ----------
    data : array, list, or tuple
        Data to be checked if pyramid or if needs to be turned into a pyramid.
    is_pyramid : bool, optional
        Value that sets if current data is considered as a pyramid or not, if
        not provided then computed.
    force_pyramid : bool or list of bool, optional
        Value that can force data to be considered as a pyramid if is_pyramid
        is not specified. If list of bool is provided then pyramiding will be
        done along axes marked as `True`. If `True` then axes that need
        pyramiding will be computed.

    Returns
    -------
    data_pyramid : list or None
        If None then data is not and does not need to be a pyramid. Otherwise
        is a list of arrays where each array is a level of the pyramid.
    """
    if is_pyramid is None:
        # If is_pyramid status is not set, guess if data currently is a pyramid
        is_pyramid = guess_pyramid(data)
        if is_pyramid is False:
            # If not a pyramid ask if is being forced to be
            if np.any(force_pyramid):
                # Either use axes provided or guess axes for pyramiding
                if isinstance(force_pyramid, bool):
                    pyr_axes = should_be_pyramid(data.shape)
                else:
                    pyr_axes = force_pyramid
                if np.any(pyr_axes):
                    # Make pyramid along target axes
                    is_pyramid = True
                    data_pyramid = make_pyramid(data, pyr_axes)
                else:
                    data_pyramid = None
            else:
                data_pyramid = None
        else:
            data_pyramid = data
    elif is_pyramid:
        # Data is a pyramid
        data_pyramid = data
    else:
        # Data is not a pyramid
        data_pyramid = None

    if data_pyramid is not None:
        data_pyramid = trim_pyramid(data_pyramid)

    return data_pyramid


def fast_pyramid(data, downscale=2, max_layer=None):
    """Compute fast image pyramid.

    In the interest of speed this method subsamples, rather than downsamples,
    the input image.

    Parameters
    ----------
    data : array
        Data from which pyramid is to be generated.
    downscale : int or list
        Factor to downscale each step of the pyramid by. If a list, one value
        must be provided for every axis of the array.
    max_layer : int, optional
        The maximum number of layers of the pyramid to be created.

    Returns
    -------
    pyramid : list
        List of arrays where each array is a level of the generated pyramid.
    """

    if max_layer is None:
        max_layer = np.floor(np.log2(np.max(data.shape))).astype(int) + 1

    zoom_factor = np.divide(1, downscale)

    pyramid = [data]
    for i in range(max_layer - 1):
        pyramid.append(
            ndi.zoom(pyramid[i], zoom_factor, prefilter=False, order=0)
        )
    return pyramid
