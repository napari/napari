"""Triangulation utilities"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, inline='always')
def _calc_output_size(
    normals: np.ndarray, closed: bool, cos_miter_limit: float, bevel: bool
) -> int:
    """Calculate the size of the output arrays for the triangulation.

    Parameters
    ----------
    normals : np.ndarray
        Nx2 array of normal vectors of the path
    closed : bool
        True if shape edge is a closed path.
    cos_miter_limit : float
        Miter limit which determines when to switch from a miter join to a
        bevel join
    bevel : bool
        If True, a bevel join is always to be used.
        If False, a bevel join will only be used when the miter limit is exceeded.

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
    point_count = len(normals) * 2
    if closed:
        point_count += 2

    bevel_count = 0

    if bevel:
        bevel_count = len(normals)
        if not closed:
            bevel_count -= 2
    else:
        for i in range(1, len(normals) - 1):
            cos_angle = (
                normals[i - 1, 0] * normals[i, 0]
                + normals[i - 1, 1] * normals[i, 1]
            )
            if cos_angle < cos_miter_limit:
                # bevel
                bevel_count += 1

        if closed:
            cos_angle = (
                normals[-2, 0] * normals[-1, 0]
                + normals[-2, 1] * normals[-1, 1]
            )
            if cos_angle < cos_miter_limit:
                bevel_count += 1
            cos_angle = (
                normals[-1, 0] * normals[0, 0] + normals[-1, 1] * normals[0, 1]
            )
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
    vec1_len: float,
    vec2_len: float,
    j: int,
    cos_limit: float,
    bevel: bool,
) -> int:
    """Set the centers, offsets, and triangles for a given path.

    Parameters
    ----------
    centers : np.ndarray
        Mx2 array central coordinates of path triangles.
    offsets : np.ndarray
        Mx2 array of the offsets to the central coordinates. Offsets need to
        be scaled by the line width and then added to the centers to
        generate the actual vertices of the triangulation
    triangles : np.ndarray
        (M-2)x3 array of the indices of the vertices that will form the
        triangles of the triangulation
    vertex : np.ndarray
        The vertex of the path for which the centers,
        offsets and triangles art calculated
    vec1 : np.ndarray
        The normal vector from previous vertex to the current vertex
    vec2 : np.ndarray
        The normal vector from the current vertex to the next vertex
    vec1_len : float
        The length of the vec1 vector, used for miter limit calculation.
    vec2_len : float
        The length of the vec2 vector, used for miter limit calculation.
    j : int
        The index of position to start putting items in the centers,
        offsets and triangles arrays
    cos_limit : float
        Miter limit which determines when to switch from a miter join to a
        bevel join
    bevel : bool
        Bool which if True causes a bevel join to always be used.
        If False a bevel join will only be used when the miter limit is exceeded

    Returns
    -------
    int
        number of triangles, centers and offsets added to the arrays
    """
    centers[j] = vertex
    centers[j + 1] = vertex
    cos_angle = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    sin_angle = vec1[0] * vec2[1] - vec1[1] * vec2[0]

    if sin_angle == 0:
        miter = np.array([vec1[1], -vec1[0]], dtype=np.float32) * 0.5
    else:
        scale_factor = 1 / sin_angle
        if bevel or cos_angle < cos_limit:
            # There is a case of bevels join, and
            # there is a need to check if the miter length is not too long.
            # For performance reasons here, the miter length is estimated
            # by the inverse of the sin of the angle between the two vectors.
            # See https://github.com/napari/napari/pull/7268#user-content-bevel-cut
            estimated_len = scale_factor
            if vec1_len < vec2_len:
                if estimated_len > vec1_len:
                    scale_factor = vec1_len
                elif estimated_len < -vec1_len:
                    scale_factor = -vec1_len
            else:
                if estimated_len > vec2_len:
                    scale_factor = vec2_len
                elif estimated_len < -vec2_len:
                    scale_factor = -vec2_len

        # We use here the Intercept theorem for calculating the miter length
        # More details in PR description:
        # https://github.com/napari/napari/pull/7268#user-content-miter
        miter = (vec1 - vec2) * 0.5 * scale_factor

    if bevel or cos_limit > cos_angle:
        centers[j + 2] = vertex
        # clock-wise and counter clock-wise cases
        if sin_angle < 0:
            offsets[j] = miter
            offsets[j + 1, 0] = -vec1[1] * 0.5
            offsets[j + 1, 1] = vec1[0] * 0.5
            offsets[j + 2, 0] = -vec2[1] * 0.5
            offsets[j + 2, 1] = vec2[0] * 0.5
            triangles[j + 1] = [j, j + 2, j + 3]
            triangles[j + 2] = [j + 2, j + 3, j + 4]
        else:
            offsets[j, 0] = vec1[1] * 0.5
            offsets[j, 1] = -vec1[0] * 0.5
            offsets[j + 1] = -miter
            offsets[j + 2, 0] = vec2[1] * 0.5
            offsets[j + 2, 1] = -vec2[0] * 0.5
            triangles[j + 1] = [j + 1, j + 2, j + 3]
            triangles[j + 2] = [j + 1, j + 3, j + 4]

        triangles[j] = [j, j + 1, j + 2]
        return 3  # bevel join added 3 triangles
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
def _orientation(p1, p2, p3) -> float:
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
def _normal_vec_and_length(
    path: np.ndarray, closed: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the normal vector and its length.

    Parameters
    ----------
    path : np.ndarray
        Mx2 array representing path to calculate the normal vector and length.
        Assumes that there is no point repetition in the path.
        In other words, assume no two consecutive points are the same.
    closed : bool
        Bool if shape edge is a closed path or not.

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
    for i in range(1, len(path)):
        vec_diff = path[i] - path[i - 1]
        vec_len_arr[i - 1] = np.sqrt(vec_diff[0] ** 2 + vec_diff[1] ** 2)
        normals[i - 1] = vec_diff / vec_len_arr[i - 1]

    if closed:
        vec_diff = path[0] - path[-1]
        vec_len_arr[-1] = np.sqrt(vec_diff[0] ** 2 + vec_diff[1] ** 2)
        normals[-1] = vec_diff / vec_len_arr[-1]

    return normals, vec_len_arr * 0.5


@njit(cache=True, inline='always')
def _generate_2D_edge_meshes_loop(
    path: np.ndarray,
    closed: bool,
    cos_limit: float,
    bevel: bool,
    normals: np.ndarray,
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
    normals : np.ndarray
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
    if closed:
        j = _set_centers_and_offsets(
            centers,
            offsets,
            triangles,
            path[0],
            normals[-1],
            normals[0],
            bevel_limit_array[-1],
            bevel_limit_array[0],
            0,
            cos_limit,
            bevel,
        )
    else:
        centers[0] = path[0]
        centers[1] = path[0]
        offsets[0, 0] = normals[0][1] * 0.5
        offsets[0, 1] = -normals[0][0] * 0.5
        offsets[1] = -offsets[0]
        triangles[0] = [0, 1, 2]
        triangles[1] = [1, 2, 3]
        j = 2

    for i in range(1, len(normals) - 1):
        j += _set_centers_and_offsets(
            centers,
            offsets,
            triangles,
            path[i],
            normals[i - 1],
            normals[i],
            bevel_limit_array[i - 1],
            bevel_limit_array[i],
            j,
            cos_limit,
            bevel,
        )
    if closed:
        j += _set_centers_and_offsets(
            centers,
            offsets,
            triangles,
            path[-1],
            normals[-2],
            normals[-1],
            bevel_limit_array[-2],
            bevel_limit_array[-1],
            j,
            cos_limit,
            bevel,
        )
        centers[j] = centers[0]
        centers[j + 1] = centers[1]
        offsets[j] = offsets[0]
        offsets[j + 1] = offsets[1]
    else:
        centers[j] = path[-1]
        centers[j + 1] = path[-1]
        offsets[j, 0] = normals[-2][1] * 0.5
        offsets[j, 1] = -normals[-2][0] * 0.5
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

    normals, bevel_limit_array = _normal_vec_and_length(path, closed)

    point_count = _calc_output_size(normals, closed, cos_limit, bevel)

    centers = np.empty((point_count, 2), dtype=np.float32)
    offsets = np.empty((point_count, 2), dtype=np.float32)
    triangles = np.empty((point_count - 2, 3), dtype=np.int32)

    _generate_2D_edge_meshes_loop(
        path,
        closed,
        cos_limit,
        bevel,
        normals,
        bevel_limit_array,
        centers,
        offsets,
        triangles,
    )

    # We fix triangle orientation to improve checks for
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
    # We would would like to use len(path) - 1 as the range.
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
