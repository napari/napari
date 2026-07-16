"""Triangulation utilities"""

from __future__ import annotations

from enum import Enum
from math import atan2, pi
from typing import Literal, overload

import numpy as np
from numba import njit
from numba.core import types
from numba.typed import List

from napari.layers.shapes.shape_types import (
    CoordinateArray,
    CoordinateArray2D,
    CoordinateArray3D,
    EdgeArray,
)


class Orientation(Enum):
    """Orientation of a triangle.

    The terms assume napari's preferred coordinate frame, in which
    the 0th axis, y, is pointing down, and the 1st axis, x, is pointing
    right. If one of the axes is flipped, the observed orientation
    would also flip.
    """

    clockwise = -1
    collinear = 0
    anticlockwise = 1


@njit(cache=True, inline='always')
def _dot(v0: np.ndarray, v1: np.ndarray) -> float:
    """Return the dot product of two 2D vectors.

    If the vectors have norm 1, this is the cosine of the angle between them.
    """
    return v0[0] * v1[0] + v0[1] * v1[1]


@njit(cache=True, inline='always')
def _cross_z(v0: np.ndarray, v1: np.ndarray) -> float:
    """Return the z-magnitude of the cross-product between two vectors.

    If the vectors have norm 1, this is the sine of the angle between them.
    """
    return v0[0] * v1[1] - v0[1] * v1[0]


@njit(cache=True, inline='always')
def _sign_abs(f: float) -> tuple[float, float]:
    """Return 1, -1, or 0 based on the sign of f, as well as the abs of f.

    The order of if-statements shows what the function is optimized for given
    the early returns. In this case, an array with many positive values will
    execute more quickly than one with many negative values.
    """
    if f > 0:
        return 1.0, f
    if f < 0:
        return -1.0, -f
    return 0.0, 0.0


@njit(cache=True, inline='always')
def _orthogonal_vector(v: np.ndarray, ccw: bool = True) -> np.ndarray:
    """Return an orthogonal vector to v (2D).

    Parameters
    ----------
    v : np.ndarray of float, shape (2,)
        The input vector.
    ccw : bool
        Whether you want the orthogonal vector in the counterclockwise
        direction (in the napari reference frame) or clockwise.

    Returns
    -------
    np.ndarray, shape (2,)
        A vector orthogonal to the input vector, and of the same magnitude.
    """
    if ccw:
        return np.array([v[1], -v[0]])
    return np.array([-v[1], v[0]])


@njit(cache=True, inline='always')
def _calc_output_size(
    direction_vectors: np.ndarray,
    closed: bool,
    cos_miter_limit: float,
    bevel: bool,
) -> int:
    """Calculate the size of the output arrays for the triangulation.

    Parameters
    ----------
    direction_vectors : np.ndarray
        Nx2 array of direction vectors along the path. The direction vectors
        have norm 1, so computing the cosine between them is just the dot
        product.
    closed : bool
        True if shape edge is a closed path.
    cos_miter_limit : float
        Miter limit which determines when to switch from a miter join to a
        bevel join
    bevel : bool
        If True, a bevel join is always used.
        If False, a bevel join will be used when the miter limit is exceeded.

    Returns
    -------
    int
        number of points in the output arrays

    Notes
    -----
    We use a cosine miter limit instead of maximum miter length for performance
    reasons. The cosine miter limit is related to the miter length by the
    following equation:

    .. math::

        c = \frac{1}{2 (l/2)^2} - 1 = \frac{2}{l^2} - 1
    """
    n = len(direction_vectors)
    # for every miter join, we have two points on either side of the path
    # vertex; if the path is closed, we add two more points — repeats of the
    # start of the path
    point_count = 2 * n + 2 * closed

    if bevel:
        # if every join is a bevel join, the computation is trivial — we need
        # one more point at each path vertex, removing the first and last
        # vertices if the path is not closed...
        bevel_count = n - 2 * (not closed)
        # ... and we can return early
        return point_count + bevel_count

    # Otherwise, we use the norm-1 direction vectors to quickly check the
    # cosine of each angle in the path. If the angle is too sharp, we get a
    # negative cosine greater than some limit, and we add a bevel point for
    # that position.
    bevel_count = 0
    # We are effectively doing a sliding window of three points along the path
    # of n points. If the path is closed, we start with indices (-1, 0, 1) and
    # end with indices (-2, -1, 0). In contrast, in an open path, we start with
    # (0, 1, 2) and end with (-3, -2, -1).
    # In the following loop, i represents the center point of the sliding
    # window. Therefore, and keeping in mind that the stop value of a range in
    # Python is exclusive, if closed we want i in the range [0, n), while if
    # open we want i in the range [1, n-1). This can be accomplished by:
    start = 1 - closed
    stop = n - 1 + closed
    for i in range(start, stop):
        cos_angle = _dot(direction_vectors[i - 1], direction_vectors[i])
        if cos_angle < cos_miter_limit:
            bevel_count += 1

    return point_count + bevel_count


@njit(cache=True, inline='always')
def _set_centers_and_offsets(
    centers: np.ndarray,
    offsets: np.ndarray,
    triangles: np.ndarray,
    vertex: np.ndarray,
    vec1: np.ndarray,
    vec2: np.ndarray,
    half_vec1_len: float,
    half_vec2_len: float,
    j: int,
    cos_limit: float,
    always_bevel: bool,
) -> int:
    """Set the centers, offsets, and triangles for a given path and position.

    This function computes the positions of the vertices of the edge triangles
    at the given path vertex and towards the next path vertex, including, if
    needed, the miter join triangle overlapping the path vertex.

    If a miter join is needed, this function will add three triangles.
    Otherwise, it will add two triangles (dividing parallelogram of the
    next edge into two triangles).

    The positions of the triangle vertices are given as the path vertex
    (repeated once for each triangle vertex), and offset vectors from that
    vertex.

    The added triangles "optimistically" index into vertices past the vertices
    added in this iteration (indices j+3 and j+4).

    Parameters
    ----------
    centers : np.ndarray of float
        Mx2 output array of central coordinates of path triangles.
    offsets : np.ndarray of float
        Mx2 output array of the offsets from the central coordinates. Offsets
        need to be scaled by the line width and then added to the centers to
        generate the actual vertices of the triangulation.
    triangles : np.ndarray of int
        (M-2)x3 output array of the indices of the vertices that will form the
        triangles of the triangulation.
    vertex : np.ndarray
        The path vertex for which the centers, offsets and triangles are
        being calculated.
    vec1 : np.ndarray
        The norm-1 direction vector from the previous path vertex to the
        current path vertex.
    vec2 : np.ndarray
        The norm-1 direction vector from the current path vertex to the next
        path vertex.
    half_vec1_len : float
        Half the length of the segment between the previous path vertex and the
        current path vertex (used for bevel join calculation).
    half_vec2_len : float
        Half the length of the segment between the current path vertex and the
        next path vertex (used for bevel join calculation).
    j : int
        The current index in the ouput arrays.
    cos_limit : float
        Miter limit which determines when to switch from a miter join to a
        bevel join, to avoid very sharp shape angles.
    always_bevel : bool
        If True, a bevel join is always used.
        If False, a bevel join will only be used when the miter limit is
        exceeded.

    Returns
    -------
    int in {2, 3}
        The number of triangles, centers and offsets added to the arrays
    """
    cos_angle = _dot(vec1, vec2)
    sin_angle = _cross_z(vec1, vec2)
    bevel = always_bevel or cos_angle < cos_limit
    for i in range(2 + bevel):
        centers[j + i] = vertex

    if sin_angle == 0:
        # if the vectors are collinear, the miter join points are exactly
        # perpendicular to the path — we can construct this vector from vec1.
        miter = np.array([vec1[1], -vec1[0]], dtype=np.float32) * 0.5
    else:
        # otherwise, we use the line intercept theorem to calculate the miter
        # direction as (vec1 - vec2) * 0.5 — these are the
        # `miter_helper_vectors` in examples/dev/triangle_edge.py — and scale.
        # If there is a bevel join, we have to make sure that the miter points
        # to the inside of the angle, *and*, we have to make sure that it does
        # not exceed the length of either incoming edge.
        # See also:
        # https://github.com/napari/napari/pull/7268#user-content-miter
        scale_factor = 1 / sin_angle
        if bevel:
            # There is a case of bevels join, and
            # there is a need to check if the miter length is not too long.
            # For performance reasons here, the miter length is estimated
            # by the inverse of the sin of the angle between the two vectors.
            # See https://github.com/napari/napari/pull/7268#user-content-bevel-cut
            sign, mag = _sign_abs(scale_factor)
            scale_factor = sign * min(mag, half_vec1_len, half_vec2_len)
        miter = (vec1 - vec2) * 0.5 * scale_factor

    if bevel:
        # add three vertices using offset vectors orthogonal to the path as
        # well as the miter vector.
        # the order in which the vertices and triangles are added depends on
        # whether the turn is clockwise or counterclockwise.
        clockwise = sin_angle < 0
        counterclockwise = not clockwise
        invert = -1.0 if counterclockwise else 1.0
        offsets[j + counterclockwise] = invert * miter
        offsets[j + clockwise] = 0.5 * _orthogonal_vector(
            vec1, ccw=counterclockwise
        )
        offsets[j + 2] = 0.5 * _orthogonal_vector(vec2, ccw=counterclockwise)
        triangles[j] = [j, j + 1, j + 2]
        triangles[j + 1] = [j + counterclockwise, j + 2, j + 3]
        triangles[j + 2] = [j + 1 + clockwise, j + 3, j + 4]

        return 3  # bevel join added 3 triangles

    # otherwise, we just use the miter vector in either direction and add two
    # triangles
    offsets[j] = miter
    offsets[j + 1] = -miter
    triangles[j] = [j, j + 1, j + 2]
    triangles[j + 1] = [j + 1, j + 2, j + 3]
    return 2  # miter join added 2 triangles


@njit(cache=True, inline='always')
def _normalize_triangle_orientation(
    triangles: np.ndarray, centers: np.ndarray, offsets: np.ndarray
) -> None:
    """Ensure vertices of all triangles are listed in the same orientation.

    In terms of napari's preferred coordinate frame (axis 0, y, is pointing
    down, axis 1, x is pointing right), this orientation is positive for
    anti-clockwise and negative for clockwise. This function normalises all
    triangle data in-place to have positive orientation.

    The orientation is useful to check if a point is inside a triangle.

    Parameters
    ----------
    triangles : np.ndarray
        (M-2)x3 array of the indices of the vertices that will form the
        triangles of the triangulation
    centers : np.ndarray
        Mx2 array central coordinates of path triangles.
    offsets : np.ndarray
        Mx2 array of the offsets to the central coordinates. Offsets need to
        be scaled by the line width and then added to the centers to
        generate the actual vertices of the triangulation
    """
    for i in range(len(triangles)):
        ti = triangles[i]
        p1 = centers[ti[0]] + offsets[ti[0]]
        p2 = centers[ti[1]] + offsets[ti[1]]
        p3 = centers[ti[2]] + offsets[ti[2]]
        if _orientation(p1, p2, p3) == Orientation.clockwise:
            triangles[i] = [ti[2], ti[1], ti[0]]


@njit(cache=True, inline='always')
def _orientation(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> Orientation:
    """Compute the orientation of three points.

    In terms of napari's preferred coordinate frame (axis 0, y, is pointing
    down, axis 1, x is pointing right), this orientation is positive for
    anti-clockwise and negative for clockwise.

    Parameters
    ----------
    p1, p2, p3: np.ndarray, shape (2,)
        The three points to check.

    Returns
    -------
    float
        Positive if anti-clockwise, negative if clockwise, 0 if collinear.
    """
    val = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    if val < 0:
        return Orientation.clockwise
    if val > 0:
        return Orientation.anticlockwise
    return Orientation.collinear


@njit(cache=True, inline='always')
def _are_polar_angles_monotonic(
    vertices: CoordinateArray2D, centroid: np.ndarray
) -> bool:
    """Check if vertices are monastical in a polar system.

    The function checks if in a polar coordinate system
    with point 0 in centroid, the angle component
    of vertices coordinate is increasing.

    This is a second part of the polygon convexity check.

    Parameters
    ----------
    vertices: np.ndarray
        Array of vertex coordinates with shape (N, 2)

    Returns
    -------
    bool
        True if the polygon is simple, False otherwise
    """
    start_angle = atan2(
        vertices[0][1] - centroid[1], vertices[0][0] - centroid[0]
    )
    prev_angle = 0.0
    for i in range(1, len(vertices)):
        angle = (
            atan2(vertices[i][1] - centroid[1], vertices[i][0] - centroid[0])
            - start_angle
        )
        if angle < 0:
            angle += 2 * pi
        if angle < prev_angle:
            return False
        prev_angle = angle
    return True


@njit(cache=True, inline='always')
def is_convex(vertices: CoordinateArray2D) -> bool:
    """Check if a polygon is convex.

    A polygon is convex when all its internal angles
    are less than or equal to 180 degrees and its edges don't
    self-intersect.

    This function determines convexity by:
    1. Checking if all non-collinear angles have the same
        orientation (clockwise or counterclockwise)
    2. Verifying that the vertices' polar angles are monotonic,
        relative to the polygon's centroid.

    - If the vertices are ordered counterclockwise, the order is
    reversed before checking.
    - Polygons with fewer than 3 vertices are not considered convex.
    - A triangle is always convex.

    Parameters
    ----------
    vertices : np.ndarray
        Array of vertex coordinates with shape (N, 2),
        where N is the number of vertices

    Returns
    -------
    bool
        True if the polygon is convex, False otherwise
    """
    if len(vertices) < 3:
        return False
    if len(vertices) == 3:
        return True
    orientation_ = Orientation.collinear
    idx = 0
    n_points = vertices.shape[0]
    orientation_set = False
    for idx in range(n_points - 2):
        p1 = vertices[idx]
        p2 = vertices[idx + 1]
        p3 = vertices[idx + 2]
        current_orientation = _orientation(p1, p2, p3)
        if current_orientation != Orientation.collinear:
            if not orientation_set:
                orientation_ = current_orientation
                orientation_set = True
            elif current_orientation != orientation_:
                return False

    if orientation_ == Orientation.collinear:
        return False

    for idx0, idx1, idx2 in [
        (n_points - 2, n_points - 1, 0),
        (n_points - 1, 0, 1),
    ]:
        triangle_orientation = _orientation(
            vertices[idx0], vertices[idx1], vertices[idx2]
        )
        if triangle_orientation not in [Orientation.collinear, orientation_]:
            return False

    centroid = np.empty(2, dtype=np.float32)
    centroid[0] = np.mean(vertices[:, 0])
    centroid[1] = np.mean(vertices[:, 1])

    if orientation_ == Orientation.anticlockwise:
        return _are_polar_angles_monotonic(vertices, centroid)

    return _are_polar_angles_monotonic(vertices[::-1], centroid)


@njit(cache=True, inline='always')
def _direction_vec_and_half_length(
    path: np.ndarray, closed: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the normal vector and its half-length.

    Parameters
    ----------
    path : np.ndarray
        Mx2 array representing path for which to calculate the direction
        vectors and the length of each segment.
        This function assumes that there is no point repetition in the path:
        consecutive points should not have identical coordinates.
    closed : bool
        True if the input path is closed (forms a loop): the edge from the
        last node in the path back to the first one is automatically inserted
        if so.

    Returns
    -------
    np.ndarray
        Mx2 array representing containing normal vectors of the path.
        If closed is False, the last vector has unspecified value.
    np.ndarray
        The array of limit of length of inner vectors in bevel joins.
        To reduce the graphical artifacts, the inner vectors are limited
        to half of the length of the adjacent egdes in path.
    """
    normals = np.empty_like(path)
    vec_len_arr = np.empty((len(path)), dtype=np.float32)
    for i in range(1 - closed, len(path)):
        vec_diff = path[i] - path[i - 1]
        vec_len_arr[i - 1] = np.sqrt(vec_diff[0] ** 2 + vec_diff[1] ** 2)
        normals[i - 1] = vec_diff / vec_len_arr[i - 1]

    return normals, vec_len_arr * 0.5


@njit(cache=True, inline='always')
def _generate_2D_edge_meshes_loop(
    path: np.ndarray,
    closed: bool,
    cos_limit: float,
    bevel: bool,
    direction_vectors: np.ndarray,
    bevel_limit_array: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray,
    triangles: np.ndarray,
) -> None:
    """Main loop for :py:func:`generate_2D_edge_meshes`.

    Parameters
    ----------
    path : np.ndarray
        Nx2 array of central coordinates of path to be triangulated
    closed : bool
        Bool which determines if the path is closed or not
    cos_limit : float
        Miter limit which determines when to switch from a miter join to a
        bevel join
    bevel : bool
        Bool which if True causes a bevel join to always be used. If False
        a bevel join will only be used when the miter limit is exceeded
    direction_vectors : np.ndarray
        Nx2 array of normal vectors of the path
    bevel_limit_array : np.ndarray
        The array of limit of length of inner vectors in bevel joins.
        To reduce the graphical artifacts, the inner vectors are limited
        to half of the length of the adjacent edges in path.
    centers : np.ndarray
        Mx2 array to put central coordinates of path triangles.
    offsets : np.ndarray
        Mx2 array to put the offsets to the central coordinates that need to
        be scaled by the line width and then added to the centers to
        generate the actual vertices of the triangulation
    triangles : np.ndarray
        (M-2)x3 array to put the indices of the vertices that will form the
        triangles of the triangulation
    """
    j = 0
    if not closed:
        centers[:2] = path[0]
        offsets[0] = 0.5 * _orthogonal_vector(direction_vectors[0], ccw=True)
        offsets[1] = -offsets[0]
        triangles[0] = [0, 1, 2]
        triangles[1] = [1, 2, 3]
        j = 2

    for i in range(1 - closed, len(direction_vectors) - 1 + closed):
        j += _set_centers_and_offsets(
            centers,
            offsets,
            triangles,
            path[i],
            direction_vectors[i - 1],
            direction_vectors[i],
            bevel_limit_array[i - 1],
            bevel_limit_array[i],
            j,
            cos_limit,
            bevel,
        )
    if closed:
        centers[j] = centers[0]
        centers[j + 1] = centers[1]
        offsets[j] = offsets[0]
        offsets[j + 1] = offsets[1]
    else:
        centers[j] = path[-1]
        centers[j + 1] = path[-1]
        offsets[j] = 0.5 * _orthogonal_vector(direction_vectors[-2])
        offsets[j + 1] = -offsets[j]


@njit(cache=True, inline='always')
def _cut_end_if_repetition(path: np.ndarray) -> np.ndarray:
    """Cut the last point of the path if it is the same as the second to last point.

    Parameters
    ----------
    path : np.ndarray
        Nx2 array of central coordinates of path to be triangulated

    Returns
    -------
    np.ndarray
        Nx2 or (N-1)x2  array of central coordinates of path to be triangulated
    """
    path_ = np.asarray(path, dtype=np.float32)
    if np.all(path_[-1] == path_[-2]):
        return path_[:-1]
    return path_


# Note: removing this decorator will double execution time.
@njit(cache=True)
def generate_2D_edge_meshes(
    path: np.ndarray,
    closed: bool = False,
    limit: float = 3.0,
    bevel: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Determines the triangulation of a path in 2D.

    The resulting `offsets`
    can be multiplied by a `width` scalar and be added to the resulting
    `centers` to generate the vertices of the triangles for the triangulation,
    i.e. `vertices = centers + width*offsets`. By using the `centers` and
    `offsets` representation, the computed triangulation can be
    independent of the line width.

    Parameters
    ----------
    path : np.ndarray
        Nx2 array of central coordinates of path to be triangulated
    closed : bool
        Bool which determines if the path is closed or not
    limit : float
        Miter limit which determines when to switch from a miter join to a
        bevel join
    bevel : bool
        Bool which if True causes a bevel join to always be used. If False
        a bevel join will only be used when the miter limit is exceeded

    Returns
    -------
    centers : np.ndarray
        Mx2 array central coordinates of path triangles.
    offsets : np.ndarray
        Mx2 array of the offsets to the central coordinates that need to
        be scaled by the line width and then added to the centers to
        generate the actual vertices of the triangulation
    triangles : np.ndarray
        (M-2)x3 array of the indices of the vertices that will form the
        triangles of the triangulation
    """

    path = _cut_end_if_repetition(path)

    if len(path) < 2:
        centers = np.empty((4, 2), dtype=np.float32)
        centers[0] = path[0]
        centers[1] = path[0]
        centers[2] = path[0]
        centers[3] = path[0]
        triangles = np.empty((2, 3), dtype=np.int32)
        triangles[0] = [0, 1, 3]
        triangles[1] = [1, 3, 2]
        return (centers, np.zeros((4, 2), dtype=np.float32), triangles)

    # why cos_limit is calculated this way is explained in the note in
    # https://github.com/napari/napari/pull/7268#user-content-bevel-limit
    cos_limit = 1 / (2 * (limit / 2) ** 2) - 1.0

    direction_vectors, bevel_limit_array = _direction_vec_and_half_length(
        path, closed
    )

    point_count = _calc_output_size(
        direction_vectors, closed, cos_limit, bevel
    )

    centers = np.empty((point_count, 2), dtype=np.float32)
    offsets = np.empty((point_count, 2), dtype=np.float32)
    triangles = np.empty((point_count - 2, 3), dtype=np.int32)

    _generate_2D_edge_meshes_loop(
        path,
        closed,
        cos_limit,
        bevel,
        direction_vectors,
        bevel_limit_array,
        centers,
        offsets,
        triangles,
    )

    # We fix triangle orientation to speed up checks for
    # whether points are in triangle.
    # Without the fix, the code is not robust to orientation.
    _normalize_triangle_orientation(triangles, centers, offsets)

    return centers, offsets, triangles


@njit(cache=True)
def remove_path_duplicates(path: np.ndarray, closed: bool) -> np.ndarray:
    """Remove consecutive duplicates from a path.

    Parameters
    ----------
    path : np.ndarray
        Nx2 or Nx3 array of central coordinates of path to be deduplicated

    Returns
    -------
    np.ndarray
        Nx2 or Nx3 array of central coordinates of deduplicated path
    """
    if len(path) <= 2:
        # part of ugly hack to keep lasso tools working
        # should be removed after fixing the lasso tool
        return path

    dup_count = 0
    # We would like to use len(path) - 1 as the range.
    # To keep the lasso tool working, we need to allow
    # duplication of the last point.
    # If the lasso tool is refactored, update to use the preferred range.
    for i in range(len(path) - 2):
        if np.all(path[i] == path[i + 1]):
            dup_count += 1

    if closed and np.all(path[0] == path[-1]):
        dup_count += 1

    if dup_count == 0:
        return path

    target_len = len(path) - dup_count
    new_path = np.empty((target_len, path.shape[1]), dtype=path.dtype)
    new_path[0] = path[0]
    index = 0
    for i in range(1, len(path)):
        if index == target_len - 1:
            break
        if np.any(new_path[index] != path[i]):
            new_path[index + 1] = path[i]
            index += 1

    return new_path


@njit(cache=True)
def create_box_from_bounding(bounding_box: np.ndarray) -> np.ndarray:
    """Creates the axis aligned interaction box of a bounding box

    Parameters
    ----------
    bounding_box : np.ndarray
        2x2 array of the bounding box. The first row is the minimum values and
        the second row is the maximum values

    Returns
    -------
    box : np.ndarray
        9x2 array of vertices of the interaction box. The first 8 points are
        the corners and midpoints of the box in clockwise order starting in the
        upper-left corner. The last point is the center of the box
    """
    x_min = bounding_box[0, 0]
    x_max = bounding_box[1, 0]
    y_min = bounding_box[0, 1]
    y_max = bounding_box[1, 1]
    result = np.empty((9, 2), dtype=np.float32)
    result[0] = [x_min, y_min]
    result[1] = [(x_min + x_max) / 2, y_min]
    result[2] = [x_max, y_min]
    result[3] = [x_max, (y_min + y_max) / 2]
    result[4] = [x_max, y_max]
    result[5] = [(x_min + x_max) / 2, y_max]
    result[6] = [x_min, y_max]
    result[7] = [x_min, (y_min + y_max) / 2]
    result[8] = [(x_min + x_max) / 2, (y_min + y_max) / 2]
    return result


@overload
def reconstruct_polygons_from_edges(
    vertices: CoordinateArray2D, edges: EdgeArray
) -> list[CoordinateArray2D]: ...


@overload
def reconstruct_polygons_from_edges(
    vertices: CoordinateArray3D, edges: EdgeArray
) -> list[CoordinateArray3D]: ...


@njit(cache=True)
def reconstruct_polygons_from_edges(
    vertices: CoordinateArray, edges: EdgeArray
) -> list[CoordinateArray2D] | list[CoordinateArray3D]:
    """Reconstruct polygons from vertices and edges.

    This function takes the output from `normalize_vertices_and_edges` — which
    is a vertex set and a list of possibly-disjoint edges — and produces a list
    of independent polygons.

    The algorithm reconstructs sub polygons using recursion.
    Starting from the first edge, it traverses the graph until it reaches starting vertex.
    After it, the algorithm iterates until reach the first non visited edge.
    So the implementation has O(M) complexity.

    Parameters
    ----------
    vertices : np.ndarray
        Array of vertex coordinates with shape (N, 2) or (N, 3)
        Cannot contain repeated vertices.
    edges : np.ndarray
        Array of edge indices with shape (M, 2)
        Cannot contain repeated edges.


    Returns
    -------
    list of np.ndarray
        List of polygons, where each polygon is an array of vertex coordinates
    """
    n_edges = edges.shape[0]
    visited = np.zeros(n_edges, dtype=np.bool_)
    n_vertices = vertices.shape[0]

    # Create a list (indexed by vertex) containing lists of incident edge indices.
    incident = List()
    for _ in range(n_vertices):
        incident.append(List.empty_list(types.int64))

    for i in range(n_edges):
        start = edges[i, 0]
        end = edges[i, 1]
        incident[start].append(i)
        incident[end].append(i)

    # List to hold all reconstructed polygons.
    polygons = List()

    for i in range(n_edges):
        if visited[i]:
            continue
        poly: list[int] = []
        # Start a new polygon with the current edge.
        start_v = edges[i, 0]
        current_v = edges[i, 1]
        poly.append(start_v)
        poly.append(current_v)
        visited[i] = True
        closed = False

        # Walk along the polygon edges until we loop back to start_v.
        while not closed:
            found = False
            # Loop through edges incident to current_v.
            for edge_idx in incident[current_v]:
                if visited[edge_idx]:
                    continue
                a = edges[edge_idx, 0]
                b = edges[edge_idx, 1]
                # Choose the vertex that is not the current one.
                next_v = a if b == current_v else b
                visited[edge_idx] = True
                if next_v == start_v:
                    closed = True
                else:
                    poly.append(next_v)
                current_v = next_v
                found = True
                if current_v == start_v:
                    closed = True
                break  # move on as soon as we find the next edge
            if not found:
                # All polygons sent as input to this function *should* be
                # closed. However, if a user accidentally passes in an open
                # chain of vertices, without this break, we would enter an
                # infinite loop. Therefore, we leave it here for safety.
                break
        polygon = vertices[np.array(poly)]
        polygons.append(polygon)
    return polygons


@njit(cache=True)
def normalize_vertices_and_edges(
    vertices: CoordinateArray2D, close: bool = False
) -> tuple[
    CoordinateArray2D,
    np.ndarray[tuple[int, Literal[2]], np.dtype[np.int64]],
]:
    """Get a list of edges that must be in triangulation for a path or polygon.

    This function ensures that:

    - no nodes are repeated, as this can cause problems with triangulation
      algorithms.
    - edges that appear twice are discarded. This allows representation of
      polygons with holes in them.

    Parameters
    ----------
    vertices: np.ndarray[np.floating], shape (N, 2)
        The 2D coordinates of the polygon's vertices. They are expected to be
        in the order in which they appear in the polygon: that is, vertices
        that follow each other are expected to be connected to each other. The
        exception is if the same edge is visited twice (in any direction): such
        edges are removed.

        Holes are expected to be represented by a polygon embedded within the
        larger polygon but winding in the opposite direction.

    close: bool
        Whether to close the polygon or treat it as a path. Note: this argument
        has no effect if the last vertex is equal to the first one — then the
        closing is explicit.

    Returns
    -------
    new_vertices: np.ndarray[np.floating], shape (M, 2)
        Vertices with duplicate nodes removed.
    edges: np.ndarray[np.intp], shape (P, 2)
        List of edges in the polygon, expressed as an array of pairs of vertex
        indices. This is usually [(0, 1), (1, 2), ... (N-1, 0)], but edges
        that are visited twice are removed.
    """
    if (
        vertices[0, 0] == vertices[-1, 0] and vertices[0, 1] == vertices[-1, 1]
    ):  # closed polygon
        vertices = vertices[:-1]  # make closing implicit
        close = True
    # Now, we make sure the vertices are unique (repeated vertices cause
    # problems in spatial algorithms, and those problems can manifest as
    # segfaults if said algorithms are implemented in C-like languages.)
    vertex_to_idx: dict[tuple[float, float], int] = {}
    new_vertices = []
    edges: set[tuple[int, int]] = set()
    prev_idx = 0
    i = 0
    for vertex in vertices:
        vertex_t = (vertex[0], vertex[1])
        if vertex_t in vertex_to_idx:
            current_idx = vertex_to_idx[vertex_t]
        else:
            current_idx = i
            vertex_to_idx[vertex_t] = i
            new_vertices.append(vertex)
            i += 1

        if prev_idx < current_idx:
            edge = (prev_idx, current_idx)
        else:
            edge = (current_idx, prev_idx)
        if edge in edges:
            edges.remove(edge)
        else:
            edges.add(edge)
        prev_idx = current_idx

    if close:
        vertex_t = (vertices[-1, 0], vertices[-1, 1])
        idx = vertex_to_idx[vertex_t]
        edge = (0, idx)
        if edge in edges:
            edges.remove(edge)
        else:
            edges.add(edge)

    edges.remove((0, 0))
    new_vertices_array = np.empty((len(new_vertices), 2), dtype=np.float32)
    for i, vertex in enumerate(new_vertices):
        new_vertices_array[i] = vertex
    edges_array = np.array(list(edges), dtype=np.int64)
    return new_vertices_array, edges_array  # type: ignore[return-value]
