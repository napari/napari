from typing import List, Optional, Tuple

import numpy as np

from ...utils.geometry import project_points_onto_plane
from ...utils.translations import trans


def _create_box_3d(
    corners: np.ndarray, normal_vector: np.ndarray, up_vector: np.ndarray
) -> np.ndarray:
    horizontal_vector = np.cross(normal_vector, up_vector)

    diagonal_vector = corners[1] - corners[0]

    up_displacement = np.dot(diagonal_vector, up_vector) * up_vector
    horizontal_displacement = (
        np.dot(diagonal_vector, horizontal_vector) * horizontal_vector
    )

    corner_1 = corners[0] + horizontal_displacement
    corner_3 = corners[0] + up_displacement

    box = np.array([corners[0], corner_1, corners[1], corner_3])
    return box


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


def create_box_from_corners_3d(
    corners: np.ndarray, normal: np.ndarray
) -> np.ndarray:
    """Create a box from two opposing corners and the normal."""
    centroid = np.mean(corners, axis=0)
    half_diagonal_vector = corners[0] - centroid
    half_diagonal_half_distance = np.linalg.norm(half_diagonal_vector)
    diagonal_unit_vector = half_diagonal_vector / half_diagonal_half_distance

    orthogonal_diagonal_vector = np.cross(diagonal_unit_vector, normal)
    corner_1 = (
        centroid + half_diagonal_half_distance * orthogonal_diagonal_vector
    )
    corner_3 = (
        centroid - half_diagonal_half_distance * orthogonal_diagonal_vector
    )

    box = np.vstack([corners[0], corner_1, corners[1], corner_3])

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
            points + 0.5 * np.array([sizes, sizes]).T,
            points + 0.5 * np.array([sizes, -sizes]).T,
            points + 0.5 * np.array([-sizes, sizes]).T,
            points + 0.5 * np.array([-sizes, -sizes]).T,
        ],
        axis=0,
    )
    return rect


def _points_in_box_3d(
    corners: np.ndarray,
    points: np.ndarray,
    sizes: np.ndarray,
    view_direction: np.ndarray,
    up_direction: np.ndarray,
) -> List[int]:
    bbox_corners = _create_box_3d(corners, view_direction, up_direction)

    # project points onto the same plane as the box
    projected_points, _ = project_points_onto_plane(
        points=points,
        plane_point=bbox_corners[0],
        plane_normal=view_direction,
    )

    horz_direction = np.cross(view_direction, up_direction)
    plane_basis = np.column_stack(
        [up_direction, horz_direction, view_direction]
    )

    bbox_corners_axis_aligned = bbox_corners @ plane_basis
    bbox_corners_axis_aligned = bbox_corners_axis_aligned[:, :2]
    points_axis_aligned = projected_points @ plane_basis
    points_axis_aligned = points_axis_aligned[:, :2]

    inside = points_in_box(
        bbox_corners_axis_aligned, points_axis_aligned, sizes
    )

    return inside


def points_in_box(
    corners: np.ndarray, points: np.ndarray, sizes: np.ndarray
) -> List[int]:
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
