"""Miscellaneous utility functions.
"""
from enum import Enum
import re
import numpy as np
import inspect
import itertools


def str_to_rgb(arg):
    """Convert an rgb string 'rgb(x,y,z)' to a list of ints [x,y,z].
    """
    return list(
        map(int, re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', arg).groups())
    )


def ensure_iterable(arg, color=False):
    """Ensure an argument is an iterable. Useful when an input argument
    can either be a single value or a list. If a color is passed then it
    will be treated specially to determine if it is iterable.
    """
    if is_iterable(arg, color=color):
        return arg
    else:
        return itertools.repeat(arg)


def has_clims(arg):
    """Check if a layer has clims.
    """
    if hasattr(arg, '_qt_controls'):
        if hasattr(arg._qt_controls, 'climSlider'):
            return True
        else:
            return False
    else:
        return False


def is_iterable(arg, color=False):
    """Determine if a single argument is an iterable. If a color is being
    provided and the argument is a 1-D array of length 3 or 4 then the input
    is taken to not be iterable.
    """
    if type(arg) is str:
        return False
    elif np.isscalar(arg):
        return False
    elif color and isinstance(arg, (list, np.ndarray)):
        if np.array(arg).ndim == 1 and (len(arg) == 3 or len(arg) == 4):
            return False
        else:
            return True
    else:
        return True


def is_multichannel(meta):
    """Determines if an image is RGB after checking its metadata.
    """
    try:
        return meta['itype'] in ('rgb', 'rgba', 'multi', 'multichannel')
    except KeyError:
        return False


def guess_multichannel(shape):
    """If last dim is 3 or 4 assume image is multichannel.
    """
    last_dim = shape[-1]

    if last_dim in (3, 4):
        return True
    else:
        return False


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

    max_shape = [0] * max_dims

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


def formatdoc(obj):
    """Substitute globals and locals into an object's docstring."""
    frame = inspect.currentframe().f_back
    try:
        obj.__doc__ = obj.__doc__.format(
            **{**frame.f_globals, **frame.f_locals}
        )
        return obj
    finally:
        del frame


def segment_normal(a, b):
    """Determines the unit normal of the vector from a to b.

    Parameters
    ----------
    a : np.ndarray
        Length 2 array of first point or Nx2 array of points
    b : np.ndarray
        Length 2 array of second point or Nx2 array of points

    Returns
    -------
    unit_norm : np.ndarray
        Length the unit normal of the vector from a to b. If a == b,
        then returns [0, 0] or Nx2 array of vectors
    """
    d = b - a

    if d.ndim == 1:
        normal = np.array([d[1], -d[0]])
        norm = np.linalg.norm(normal)
        if norm == 0:
            norm = 1
    else:
        normal = np.stack([d[:, 1], -d[:, 0]], axis=0).transpose(1, 0)
        norm = np.linalg.norm(normal, axis=1, keepdims=True)
        ind = norm == 0
        norm[ind] = 1
    unit_norm = normal / norm

    return unit_norm


def segment_normal_vector(a, b):
    """Determines the unit normal of the vector from a to b.

    Parameters
    ----------
    a : np.ndarray
        Length 2 array of first point
    b : np.ndarray
        Length 2 array of second point

    Returns
    -------
    unit_norm : np.ndarray
        Length the unit normal of the vector from a to b. If a == b,
        then returns [0, 0]
    """
    d = b - a
    normal = np.array([d[1], -d[0]])
    norm = np.linalg.norm(normal)
    if norm == 0:
        unit_norm = np.array([0, 0])
    else:
        unit_norm = normal / norm
    return unit_norm


class StringEnum(Enum):
    def _generate_next_value_(name, start, count, last_values):
        """ autonaming function assigns each value its own name as a value
        """
        return name.lower()

    def _missing_(self, value):
        """ function called with provided value does not match any of the class
           member values. This function tries again with an upper case string.
        """
        return self(value.lower())

    def __str__(self):
        """String representation: The string method returns the
        valid vispy symbol string for the Markers visual.
        """
        return self.value
