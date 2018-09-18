"""Miscellaneous utility functions.
"""
from vispy import scene as _scene


# get available interpolation methods
interpolation_names = _scene.visuals.Image(None).interpolation_functions
interpolation_names = list(interpolation_names)
interpolation_names.sort()
# interpolation_names.remove('sinc')  # does not work well on my machine

interpolation_index_to_name = interpolation_names.__getitem__
interpolation_name_to_index = interpolation_names.index


def is_multichannel(meta):
    """Determines if an image is RGB after checking its metadata.
    """
    try:
        return meta['itype'] in ('rgb', 'rgba' 'multi', 'multichannel')
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

    max_shape = [0,] * max_dims

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
_windows = []
_GUESS = object()


def imshow(image, meta=None, multichannel=_GUESS,
           new_window=True, **kwargs):
    """Displays an image.

    Parameters
    ----------
    image : np.ndarray
        Image data.
    meta : dict, optional
        Image metadata.
    multichannel : bool, optional
        Whether the image is multichannel.
    new_window : bool, optional
        Whether the image will open in a new window.
    **kwargs : dict
        Parameters that will be translated to metadata.

    Returns
    -------
    container : ImageContainer
        Image container.
    """
    from PyQt5.QtWidgets import QApplication
    from .elements.image_window import ImageWindow

    if isinstance(meta, dict):
        meta = dict(meta, **kwargs)

    if meta is None:
        meta = kwargs

    if multichannel is _GUESS:
        multichannel = guess_multichannel(image.shape)

    if multichannel:
        meta['itype'] = 'multi'

    global _app
    _app = QApplication.instance() or QApplication([])

    if not _windows or new_window:
        window = ImageWindow()
        _windows.append(window)
    else:
        window = _windows[-1]

    container = window.add_viewer().add_image(image, meta)

    window.show()
    window.raise_()

    return container
