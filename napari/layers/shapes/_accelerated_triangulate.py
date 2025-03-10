"""Triangulation utilities"""

from __future__ import annotations

import numpy as np
from numba import njit


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
        if _orientation(p1, p2, p3) < 0:
            triangles[i] = [ti[2], ti[1], ti[0]]


@njit(cache=True, inline='always')
def _orientation(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
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
    # fmt: off
    return (
        (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    )
    # fmt: on


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
