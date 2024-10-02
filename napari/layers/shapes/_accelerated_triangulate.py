from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _calc_output_size(
    normals: np.ndarray, closed: bool, cos_limit: float, bevel: bool
) -> int:
    """Calculate the size of the output arrays for the triangulation.

    Parameters
    ----------
    normals : np.ndarray
        Nx2 array of normal vectors of the path
    closed : bool
        Bool if shape edge is a closed path or not.
    cos_limit : float
        Miter limit which determines when to switch from a miter join to a
        bevel join
    bevel : bool
        Bool which if True causes a bevel join to always be used.
        If False a bevel join will only be used when the miter limit is exceeded

    Returns
    -------
    int
        number of points in the output arrays

    Notes
    -----
    we use cos_limit instead of maximum miter length
    for performance reasons.
    This is an equivalent check, see note in generate_2D_edge_meshes
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
            if cos_limit > cos_angle:
                # bevel
                bevel_count += 1

        if closed:
            cos_angle = (
                normals[-2, 0] * normals[-1, 0]
                + normals[-2, 1] * normals[-1, 1]
            )
            if cos_limit > cos_angle:
                bevel_count += 1
            cos_angle = (
                normals[-1, 0] * normals[0, 0] + normals[-1, 1] * normals[0, 1]
            )
            if cos_limit > cos_angle:
                bevel_count += 1

    return point_count + bevel_count


@njit(cache=True)
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
        Mx2 array of the offsets to the central coordinates that need to
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
        mitter = np.array([vec1[1], -vec1[0]], dtype=np.float32) * 0.5
    else:
        if bevel or cos_limit > cos_angle:
            # There is a case of bevels join, and
            # there is a need to check if the miter length is not too long.
            # For performance reasons here, the mitter length is estimated
            # by the inverse of the sin of the angle between the two vectors.
            elapsed_len = 1 / sin_angle
            if vec1_len < vec2_len:
                if elapsed_len > vec1_len:
                    sin_angle = 1 / vec1_len
                elif elapsed_len < -vec1_len:
                    sin_angle = -1 / vec1_len
            else:
                if elapsed_len > vec2_len:
                    sin_angle = 1 / vec2_len
                elif elapsed_len < -vec2_len:
                    sin_angle = -1 / vec2_len


        # We use here the Intercept theorem for calculating the mitter length
        mitter = (vec1 - vec2) * 0.5 * (1 / sin_angle)
    if bevel or cos_limit > cos_angle:
        centers[j + 2] = vertex
        # clock-wise and counter clock-wise cases
        if sin_angle < 0:
            offsets[j] = mitter
            offsets[j + 1, 0] = -vec1[1] * 0.5
            offsets[j + 1, 1] = vec1[0] * 0.5
            offsets[j + 2, 0] = -vec2[1] * 0.5
            offsets[j + 2, 1] = vec2[0] * 0.5
            triangles[j + 1] = [j, j + 2, j + 3]
            triangles[j + 2] = [j + 2, j + 3, j + 4]
        else:
            offsets[j, 0] = vec1[1] * 0.5
            offsets[j, 1] = -vec1[0] * 0.5
            offsets[j + 1] = -mitter
            offsets[j + 2, 0] = vec2[1] * 0.5
            offsets[j + 2, 1] = -vec2[0] * 0.5
            triangles[j + 1] = [j + 1, j + 2, j + 3]
            triangles[j + 2] = [j + 1, j + 3, j + 4]

        triangles[j] = [j, j + 1, j + 2]

        return 3  # added 3 triangles because of bevel
    offsets[j] = mitter
    offsets[j + 1] = -mitter
    triangles[j] = [j, j + 1, j + 2]
    triangles[j + 1] = [j + 1, j + 2, j + 3]
    return 2  # added 2 triangles


@njit(cache=True)
def _fix_triangle_orientation(
    triangles: np.ndarray, centers: np.ndarray, offsets: np.ndarray
) -> None:
    """Fix the orientation of the triangles.

    For checking if a point is inside a triangle.

    Parameters
    ----------
    triangles : np.ndarray
        (M-2)x3 array of the indices of the vertices that will form the
        triangles of the triangulation
    centers : np.ndarray
        Mx2 array central coordinates of path triangles.
    offsets : np.ndarray
        Mx2 array of the offsets to the central coordinates that need to
        be scaled by the line width and then added to the centers to
        generate the actual vertices of the triangulation
    """
    for i in range(len(triangles)):
        triangle = triangles[i]
        p1 = centers[triangle[0]] + offsets[triangle[0]]
        p2 = centers[triangle[1]] + offsets[triangle[1]]
        p3 = centers[triangle[2]] + offsets[triangle[2]]
        if (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (
            p3[0] - p1[0]
        ) < 0:
            triangles[i] = [triangle[2], triangle[1], triangle[0]]


@njit(cache=True)
def generate_2D_edge_meshes(
    path: np.ndarray,
    closed: bool = False,
    limit: float = 3.0,
    bevel: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Determines the triangulation of a path in 2D. The resulting `offsets`
    can be multiplied by a `width` scalar and be added to the resulting
    `centers` to generate the vertices of the triangles for the triangulation,
    i.e. `vertices = centers + width*offsets`. Using the `centers` and
    `offsets` representation thus allows for the computed triangulation to be
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

    path = np.asarray(path, dtype=np.float32)

    if np.all(path[-1] == path[-2]):
        # Part of ugly hack to keep lasso tools working
        # should be removed after fixing the lasso tool
        path = path[:-1]

    if len(path) <= 2:
        centers = np.empty((4, 2), dtype=np.float32)
        centers[0] = path[0]
        centers[1] = path[0]
        centers[2] = path[0]
        centers[3] = path[0]
        triangles = np.empty((2, 3), dtype=np.int32)
        triangles[0] = [0, 1, 3]
        triangles[1] = [1, 3, 2]
        return (centers, np.zeros((4, 2), dtype=np.float32), triangles)

    cos_limit = -np.float32(
        np.sqrt(1.0 - 1.0 / ((limit / 2) ** 2))
    )  # divide by 2 to be consistent with the original code

    normals = np.empty_like(path)
    vec_len_arr = np.empty((len(path) - 1), dtype=np.float32)
    for i in range(1, len(path)):
        vec_diff = path[i] - path[i - 1]
        vec_len_arr[i - 1] = np.sqrt(vec_diff[0] ** 2 + vec_diff[1] ** 2)
        normals[i - 1] = vec_diff / vec_len_arr[i - 1]

    if closed:
        vec_diff = path[0] - path[-1]
        vec_len_arr[-1] = np.sqrt(vec_diff[0] ** 2 + vec_diff[1] ** 2)
        normals[-1] = vec_diff / vec_len_arr[-1]

    point_count = _calc_output_size(normals, closed, cos_limit, bevel)

    centers = np.empty((point_count, 2), dtype=np.float32)
    offsets = np.empty((point_count, 2), dtype=np.float32)
    triangles = np.empty((point_count - 2, 3), dtype=np.int32)

    if closed:
        j = _set_centers_and_offsets(
            centers,
            offsets,
            triangles,
            path[0],
            normals[-1],
            normals[0],
            vec_len_arr[-1],
            vec_len_arr[0],
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
            vec_len_arr[i - 1],
            vec_len_arr[i],
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
            vec_len_arr[-2],
            vec_len_arr[-1],
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

    # We need to fix triangle orientation, as our code for checking
    # if points is in triangle is not robust to orientation
    _fix_triangle_orientation(triangles, centers, offsets)

    return centers, offsets, triangles


@njit(cache=True)
def remove_path_duplicates(path: np.ndarray, closed: bool) -> np.ndarray:
    """Remove duplicates from a path.

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
    for i in range(len(path) - 2):
        # should be len(path) - 1 but we need to allow
        # duplication of the last point to keep the lasse tool working
        if np.all(path[i] == path[i + 1]):
            dup_count += 1

    if closed and np.all(path[0] == path[-1]):
        dup_count += 1

    if dup_count == 0:
        return path

    target_len = len(path) - dup_count
    new_path = np.empty((target_len, path.shape[1]), dtype=np.float32)
    index = 0

    new_path[0] = path[0]
    for i in range(1, len(path)):
        if index == target_len - 1:
            break
        if np.any(new_path[index] != path[i]):
            new_path[index + 1] = path[i]
            index += 1

    return new_path
