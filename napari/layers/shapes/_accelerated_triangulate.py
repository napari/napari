from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _calc_output_size(
    normals: np.ndarray, closed: bool, cos_limit: float, bevel: bool
) -> int:
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
    path: np.ndarray,
    vec1: np.ndarray,
    vec2: np.ndarray,
    j: int,
    cos_limit: float,
    sin_limit: float,
) -> int:
    centers[j] = path
    centers[j + 1] = path
    cos_angle = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    sin_angle = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    if sin_angle == 0:
        mitter = np.array([vec1[1], -vec1[0]], dtype=np.float32) * 0.5
    else:
        if sin_limit > sin_angle > -sin_limit and cos_limit > cos_angle:
            sin_angle = sin_limit if sin_angle > 0 else -sin_limit
        mitter = (vec1 - vec2) * 0.5 * (1 / sin_angle)
    if cos_limit > cos_angle:
        centers[j + 2] = path
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

        return 3
    offsets[j] = mitter
    offsets[j + 1] = -mitter
    triangles[j] = [j, j + 1, j + 2]
    triangles[j + 1] = [j + 1, j + 2, j + 3]
    return 2


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
        Nx2 or Nx3 array of central coordinates of path to be triangulated
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
        Mx2 or Mx3 array central coordinates of path triangles.
    offsets : np.ndarray
        Mx2 or Mx3 array of the offsets to the central coordinates that need to
        be scaled by the line width and then added to the centers to
        generate the actual vertices of the triangulation
    triangles : np.ndarray
        Px3 array of the indices of the vertices that will form the
        triangles of the triangulation
    """

    path = np.asarray(path, dtype=np.float32)

    cos_limit = -np.float32(np.sqrt(1.0 - 1.0 / ((limit /2)**2))) # divide by 2 to be consistent with the original code
    sin_limit  = 1.0 / limit  # limit the maximum length of the offset vector

    normals = np.empty_like(path)
    for i in range(1, len(path)):
        vec_diff = path[i] - path[i - 1]
        normals[i - 1] = vec_diff / np.sqrt(
            vec_diff[0] ** 2 + vec_diff[1] ** 2
        )

    if closed:
        vec_diff = path[0] - path[-1]
        normals[-1] = vec_diff / np.sqrt(vec_diff[0] ** 2 + vec_diff[1] ** 2)

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
            0,
            cos_limit,
            sin_limit,
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
            j,
            cos_limit,
            sin_limit,
        )
    if closed:
        j += _set_centers_and_offsets(
            centers,
            offsets,
            triangles,
            path[-1],
            normals[-2],
            normals[-1],
            j,
            cos_limit,
            sin_limit,
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

    return centers, offsets, triangles
