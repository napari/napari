from typing import Optional, Tuple

import numpy as np

from ...utils.translations import trans


def create_box(data):
    """Create the axis aligned interaction box of a list of points

    Parameters
    ----------
    data : (N, 2) array
        Points around which the interaction box is created

    Returns
    -------
    box : (4, 2) array
        Vertices of the interaction box
    """
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)
    tl = np.array([min_val[0], min_val[1]])
    tr = np.array([max_val[0], min_val[1]])
    br = np.array([max_val[0], max_val[1]])
    bl = np.array([min_val[0], max_val[1]])
    box = np.array([tl, tr, br, bl])
    return box


def points_to_squares(points, sizes):
    """Expand points to squares defined by their size

    Parameters
    ----------
    points : (N, 2) array
        Points to be turned into squares
    sizes : (N,) array
        Size of each point

    Returns
    -------
    rect : (4N, 2) array
        Vertices of the expanded points
    """
    rect = np.concatenate(
        [
            points + np.sqrt(2) / 2 * np.array([sizes, sizes]).T,
            points + np.sqrt(2) / 2 * np.array([sizes, -sizes]).T,
            points + np.sqrt(2) / 2 * np.array([-sizes, sizes]).T,
            points + np.sqrt(2) / 2 * np.array([-sizes, -sizes]).T,
        ],
        axis=0,
    )
    return rect


def points_in_box(corners, points, sizes):
    """Determine which points are in an axis aligned box defined by the corners

    Parameters
    ----------
    points : (N, 2) array
        Points to be checked
    sizes : (N,) array
        Size of each point

    Returns
    -------
    inside : list
        Indices of points inside the box
    """
    box = create_box(corners)[[0, 2]]
    # Check all four corners in a square around a given point. If any corner
    # is inside the box, then that point is considered inside
    point_corners = points_to_squares(points, sizes)
    below_top = np.all(box[1] >= point_corners, axis=1)
    above_bottom = np.all(point_corners >= box[0], axis=1)
    point_corners_in_box = np.where(np.logical_and(below_top, above_bottom))[0]
    # Determine indices of points which have at least one corner inside box
    inside = np.unique(point_corners_in_box % len(points))
    return list(inside)


def fix_data_points(
    points: Optional[np.ndarray], ndim: Optional[int]
) -> Tuple[np.ndarray, int]:
    """
    Ensure that points array is 2d and have second dimension of size ndim (default 2 for empty arrays)

    Parameters
    ----------
    points : (N, M) array or None
        Points to be checked
    ndim : int or None
        number of expected dimensions

    Returns
    -------
    points : (N, M) array
        Points array
    ndim : int
        number of dimensions

    Raises
    ------
    ValueError
        if ndim does not match with second dimensions of points
    """
    if points is None or len(points) == 0:
        if ndim is None:
            ndim = 2
        points = np.empty((0, ndim))
    else:
        points = np.atleast_2d(points)
        data_ndim = points.shape[1]
        if ndim is not None and ndim != data_ndim:
            raise ValueError(
                trans._(
                    "Points dimensions must be equal to ndim",
                    deferred=True,
                )
            )
        ndim = data_ndim
    return points, ndim
