"""Miscellaneous utility functions.
"""
import __main__


INTERACTIVE = not hasattr(__main__, '__file__')


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
_windows = []


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
    viewer : Viewer
        Viewer containing the image.
    """
    from ..components import Window, Viewer, QtApplication

    global _app
    _app = _app or QtApplication.instance() or QtApplication([])

    window = Window(Viewer(), show=False)
    _windows.append(window)
    layer = window.viewer.add_image(image, meta, multichannel, **kwargs)
    window.show()

    if not INTERACTIVE:
        _app.exec()

    return window.viewer


def scatter(coords, symbol='o', size=10, edge_width=1,
            edge_width_rel=None, edge_color='black', face_color='white',
            scaling=True, n_dimensional=False):
    """Displays a scatterplot with markers of the given properties.

    Parameters
    ----------
    coords : np.ndarray
        coordinates for each marker.

    symbol : str
        symbol to be used as a marker

    size : int, float, np.ndarray, list
        size of the marker. If given as a scalar, all markers are the
        same size. If given as a list/array, size must be the same
        length as coords and sets the marker size for each marker
        in coords (element-wise). If n_dimensional is True then can be a list
        of length dims or can be an array of shape Nxdims where N is the number
        of markers and dims is the number of dimensions

    edge_width : int, float, None
        width of the symbol edge in px
            vispy docs say: "exactly one edge_width and
            edge_width_rel must be supplied"

    edge_width_rel : int, float, None
        width of the marker edge as a fraction of the marker size.

            vispy docs say: "exactly one edge_width and
            edge_width_rel must be supplied"

    edge_color : Color, ColorArray
        color of the marker border

    face_color : Color, ColorArray
        color of the marker body

    scaling : bool
        if True, marker rescales when zooming

    n_dimensional : bool
        if True, renders markers not just in central plane but also in all
        n dimensions according to specified marker size

    Returns
    -------
    viewer : Viewer
        Viewer containing the markers.
    """
    from ..components import Window, Viewer, QtApplication

    global _app
    _app = _app or QtApplication.instance() or QtApplication([])

    window = Window(Viewer(), show=False)
    _windows.append(window)
    layer = window.viewer.add_markers(coords, symbol, size, edge_width,
                                      edge_width_rel, edge_color,
                                      face_color, scaling, n_dimensional)
    window.show()

    if not INTERACTIVE:
        _app.exec()

    return window.viewer
