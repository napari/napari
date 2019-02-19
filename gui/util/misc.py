"""Miscellaneous utility functions.
"""
from numpy import multiply, all, array

def inside_triangles(triangles):
    """Checks which triangles contain the origin
    Parameters
    ----------
    boxes : np.ndarray
        Nx3x2 array of N triangles that should be checked
    """

    AB = triangles[:,1,:] - triangles[:,0,:]
    AC = triangles[:,2,:] - triangles[:,0,:]
    BC = triangles[:,2,:] - triangles[:,1,:]

    s_AB = -AB[:,0]*triangles[:,0,1] + AB[:,1]*triangles[:,0,0] >= 0
    s_AC = -AC[:,0]*triangles[:,0,1] + AC[:,1]*triangles[:,0,0] >= 0
    s_BC = -BC[:,0]*triangles[:,1,1] + BC[:,1]*triangles[:,1,0] >= 0

    return all(array([s_AB != s_AC, s_AB == s_BC]), axis=0)

def inside_boxes(boxes):
    """Checks which boxes contain the origin
    Parameters
    ----------
    boxes : np.ndarray
        Nx8x2 array of N boxes that should be checked
    """

    AB = boxes[:,0] - boxes[:,6]
    AM = boxes[:,0]
    BC = boxes[:,6] - boxes[:,4]
    BM = boxes[:,6]

    ABAM = multiply(AB, AM).sum(1)
    ABAB = multiply(AB, AB).sum(1)
    BCBM = multiply(BC, BM).sum(1)
    BCBC = multiply(BC, BC).sum(1)

    c1 = 0 <= ABAM
    c2 = ABAM <= ABAB
    c3 = 0 <= BCBM
    c4 = BCBM <= BCBC

    return all(array([c1, c2, c3, c4]), axis=0)

def is_permutation(ar, N):
    """Checks is an array is a permutation of the intergers 0, ... N-1
    Parameters
    ----------
    ar : list or np.ndarray
        Array that should be checked
    N : int
        Integer defining length of target permutation
    """
    # Check if array is corret length
    if len(ar) != N:
        return False

    appears = [False for n in range(N)]

    for i in ar:
        # Check if element is in bounds
        if i < 0 or i > N-1:
            return False

        # Check if element has appeared
        if appears[i]:
            return False
        else:
            appears[i] = True

    # If successfully passed through list
    return True

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
    window: Window
        Window object.
    """
    from ..components import Window, Viewer, QtApplication

    meta = guess_metadata(image, meta, multichannel, kwargs)

    global _app
    _app = _app or QtApplication.instance() or QtApplication([])

    window = Window(Viewer(), show=False)
    _windows.append(window)
    layer = window.viewer.add_image(image, meta)
    window.show()

    return window.viewer
