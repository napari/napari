import os

from .vendored import colorconv
import numpy as np
import vispy.color


_matplotlib_list_file = os.path.join(os.path.dirname(__file__),
                                     'matplotlib_cmaps.txt')
with open(_matplotlib_list_file) as fin:
    matplotlib_colormaps = [line.rstrip() for line in fin]


def _validate_rgb(colors, *, tolerance=0.):
    """Return the subset of colors that is in [0, 1] for all channels.

    Parameters
    ----------
    colors : array of float, shape (N, 3)
        Input colors in RGB space.

    Other Parameters
    ----------------
    tolerance : float, optional
        Values outside of the range by less than ``tolerance`` are allowed and
        clipped to be within the range.

    Returns
    -------
    filtered_colors : array of float, shape (M, 3), M <= N
        The subset of colors that are in valid RGB space.

    Examples
    --------
    >>> colors = np.array([[  0. , 1.,  1.  ],
    ...                    [  1.1, 0., -0.03],
    ...                    [  1.2, 1.,  0.5 ]])
    >>> _validate_rgb(colors)
    array([[0., 1., 1.]])
    >>> _validate_rgb(colors, tolerance=0.15)
    array([[0., 1., 1.],
           [1., 0., 0.]])
    """
    lo = 0 - tolerance
    hi = 1 + tolerance
    valid = np.all((colors > lo) & (colors < hi), axis=1)
    filtered_colors = np.clip(colors[valid], 0, 1)
    return filtered_colors


def _low_discrepancy(dim, n):
    """Generate a 1d, 2d, or 3d low discrepancy sequence of coordinates.

    Parameters
    ----------
    dim : one of {1, 2, 3}
        The dimensionality of the sequence.
    n : int
        How many points to generate.

    Returns
    -------
    pts : array of float, shape (n, dim)
        The sampled points.

    References
    ----------
    ..[1]: http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    """
    phi1 = 1.6180339887498948482
    phi2 = 1.32471795724474602596
    phi3 = 1.22074408460575947536
    phi = np.array([phi1, phi2, phi3])
    g = 1 / phi
    n = np.reshape(np.arange(n), (n, 1))
    pts = (0.5 + (n * g[:dim])) % 1
    return pts


def label_colormap(labels):
    """Produce a colormap suitable for use with a given label set.

    Parameters
    ----------
    labels : array of int
        A set of labels or label image.

    Returns
    -------
    cmap : vispy.color.Colormap
        A colormap for use with ``labels``. The labels are remapped so that
        the maximum label falls on 1.0, since vispy requires colormaps to map
        within [0, 1].

    Notes
    -----
    0 always maps to fully transparent.
    """
    unique_labels = np.unique(labels)
    max_label = np.max(unique_labels)
    unique_labels_float = unique_labels / max_label
    midpoints = np.convolve(unique_labels_float, [0.5, 0.5], mode='valid')
    control_points = np.concatenate(([-np.eps], midpoints, [1+np.eps]))
