"""Miscellaneous utility functions.
"""


def is_multichannel(meta):
    """Determines if an image is RGB after checking its metadata.
    """
    try:
        return meta['itype'] in ('rgb', 'rgba', 'multi', 'multichannel')
    except KeyError:
        return False


def guess_multichannel(shape):
    """Guesses if an image is multichannel based on its shape.
    """
    first_dims = shape[:-1]
    last_dim = shape[-1]

    average = sum(first_dims) / len(first_dims)

    if average * .95 - 1 <= last_dim <= average * 1.05 + 1:
        # roughly all dims are the same
        return False

    if last_dim in (3, 4):
        if average > 10:
            return True

    diff = average - last_dim

    return diff > last_dim * 100


def guess_metadata(image, meta, multichannel, kwargs):
    """Guesses an image's metadata.

    Parameters
    ----------
    image : np.ndarray
        Image data.
    meta : dict or None
        Image metadata.
    multichannel : bool or None
        Whether the image is multichannel. Guesses if None.
    kwargs : dict
        Parameters that will be translated to metadata.

    Returns
    -------
    meta : dict
        Guessed image metadata.
    """
    if isinstance(meta, dict):
        meta = dict(meta, **kwargs)

    if meta is None:
        meta = kwargs

    if multichannel is None:
        multichannel = guess_multichannel(image.shape)

    if multichannel:
        meta['itype'] = 'multi'

    return meta


def compute_max_shape(shapes, max_dims=None):
    """Computes the maximum shape combination from the given shapes.

    Parameters
    ----------
    shapes : iterable of tuple
        Shapes to coombine.
    max_dims : int, optional
        Pre-computed maximum dimensions of the final shape.
        If None, is computed on the fly.

    Returns
    -------
    max_shape : tuple
        Maximum shape combination.
    """
    shapes = tuple(shapes)

    if max_dims is None:
        max_dims = max(len(shape) for shape in shapes)

    max_shape = [0, ] * max_dims

    for dim in range(max_dims):
        for shape in shapes:
            try:
                dim_len = shape[dim]
            except IndexError:
                pass
            else:
                if dim_len > max_shape[dim]:
                    max_shape[dim] = dim_len
    return tuple(max_shape)


_app = None


def imshow(image, meta=None, multichannel=None, **kwargs):
    """Displays an image.

    Parameters
    ----------
    image : np.ndarray
        Image data.
    meta : dict, optional
        Image metadata.
    multichannel : bool, optional
        Whether the image is multichannel. Guesses if None.
    **kwargs : dict
        Parameters that will be translated to metadata.

    Returns
    -------
    window: Window
        Window object.
    """
    from ..elements import Window, Viewer
    from ..elements.qt import QtApplication

    meta = guess_metadata(image, meta, multichannel, kwargs)

    global _app
    _app = _app or QtApplication.instance() or QtApplication([])

    window = Window(Viewer(), show=False)
    layer = window.viewer.add_image(image, meta)
    window.show()

    return window.viewer
