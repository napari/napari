"""Miscellaneous utility functions.
"""
import numpy as np
import inspect

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

    return np.all(np.array([s_AB != s_AC, s_AB == s_BC]), axis=0)

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

    ABAM = np.multiply(AB, AM).sum(1)
    ABAB = np.multiply(AB, AB).sum(1)
    BCBM = np.multiply(BC, BM).sum(1)
    BCBC = np.multiply(BC, BC).sum(1)

    c1 = 0 <= ABAM
    c2 = ABAM <= ABAB
    c3 = 0 <= BCBM
    c4 = BCBM <= BCBC

    return np.all(np.array([c1, c2, c3, c4]), axis=0)


def point_to_lines(point, lines):
    """Calculate the distance between a point and line segments. First calculates
    the distance to the infinite line, then checks if the projected point lies
    between the line segment endpoints. If not, calculates distance to the endpoints
    Parameters
    ----------
    point : np.ndarray
        1x2 array of point should be checked
    lines : np.ndarray
        Nx2x2 array of line segments
    """

    # shift and normalize vectors
    lines_vectors = lines[:,1] - lines[:,0]
    point_vectors = point - lines[:,0]
    end_point_vectors = point - lines[:,1]
    norm_lines = np.linalg.norm(lines_vectors, axis=1, keepdims=True)
    reject = (norm_lines==0).squeeze()
    norm_lines[reject] = 1
    unit_lines = lines_vectors / norm_lines

    # calculate distance to line
    line_dist = abs(np.cross(unit_lines, point_vectors))

    # calculate scale
    line_loc = (unit_lines*point_vectors).sum(axis=1)/norm_lines.squeeze()

    # for points not falling inside segment calculate distance to appropriate endpoint
    line_dist[line_loc<0] = np.linalg.norm(point_vectors[line_loc<0], axis=1)
    line_dist[line_loc>1] = np.linalg.norm(end_point_vectors[line_loc>1], axis=1)
    line_dist[reject] = np.linalg.norm(point_vectors[reject], axis=1)
    line_loc[reject] = 0.5

    # calculate closet line
    ind = np.argmin(line_dist)

    return ind, line_loc[ind]


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


def formatdoc(obj):
    """Substitute globals and locals into an object's docstring."""
    frame = inspect.currentframe().f_back
    try:
        obj.__doc__ = obj.__doc__.format(**{**frame.f_globals, **frame.f_locals})
        return obj
    finally:
        del frame
