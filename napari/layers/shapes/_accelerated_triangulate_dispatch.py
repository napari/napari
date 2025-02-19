"""Module providing a unified interface for triangulation helper functions.

We don't want numba to be a required dependency. Therefore, for
numba-accelerated functions, we provide slower NumPy-only alternatives.

With this module, downstream modules can import these helper functions without
knowing which implementation is being used.
"""

import numpy as np
import numpy.typing as npt

from napari.layers.utils.layer_utils import segment_normal

__all__ = (
    'USE_COMPILED_BACKEND',
    'create_box_from_bounding',
    'generate_2D_edge_meshes',
    'generate_2D_edge_meshes_py',
    'remove_path_duplicates',
    'warmup_numba_cache',
)


def remove_path_duplicates_np(
    data: npt.NDArray[np.float32], closed: bool
) -> npt.NDArray[np.float32]:
    # We add the first data point at the end to get the same length bool
    # array as the data, and also to work on closed shapes; the last value
    # in the diff array compares the last and first vertex.
    diff = np.diff(np.append(data, data[0:1, :], axis=0), axis=0)
    dup = np.all(diff == 0, axis=1)
    # if the shape is closed, check whether the first vertex is the same
    # as the last vertex, and count it as a duplicate if so
    if closed and dup[-1]:
        dup[0] = True
    # we allow repeated nodes at the end for the lasso tool, which
    # for an instant needs both the last placed point and the point at the
    # cursor to be the same; if the lasso implementation becomes cleaner,
    # remove this hardcoding
    dup[-2:] = False
    indices = np.arange(data.shape[0])
    return data[indices[~dup]]


def _mirror_point(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2 * y - x


def _sign_nonzero(x: np.ndarray) -> npt.NDArray[np.int32]:
    y = np.sign(x).astype(np.int32)
    y[y == 0] = 1
    return y


def _sign_cross(x: np.ndarray, y: np.ndarray) -> npt.NDArray[np.int32]:
    """sign of cross product (faster for 2d)"""
    if x.shape[1] == y.shape[1] == 2:
        return _sign_nonzero(x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0])
    if x.shape[1] == y.shape[1] == 3:
        return _sign_nonzero(np.cross(x, y))

    raise ValueError(x.shape[1], y.shape[1])


def generate_2D_edge_meshes_py(
    path: npt.NDArray[np.float32],
    closed: bool = False,
    limit: float = 3,
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

    path = np.asarray(path, dtype=float)

    # add first vertex to the end if closed
    if closed:
        path = np.concatenate((path, [path[0]]))

    # extend the path by adding a vertex at beginning and end
    # to get the mean normals correct
    if closed:
        _ext_point1 = path[-2]
        _ext_point2 = path[1]
    else:
        _ext_point1 = _mirror_point(path[1], path[0])
        _ext_point2 = _mirror_point(path[-2], path[-1])

    full_path = np.concatenate(([_ext_point1], path, [_ext_point2]), axis=0)

    # full_normals[:-1], full_normals[1:] are normals left and right of each path vertex
    full_normals = segment_normal(full_path[:-1], full_path[1:])

    # miters per vertex are the average of left and right normals
    miters = 0.5 * (full_normals[:-1] + full_normals[1:])

    # scale miters such that their dot product with normals is 1
    _mf_dot = np.expand_dims(
        np.einsum('ij,ij->i', miters, full_normals[:-1]), -1
    )

    miters = np.divide(
        miters,
        _mf_dot,
        where=np.abs(_mf_dot) > 1e-10,
    )

    miter_lengths_squared = (miters**2).sum(axis=1)

    # miter_signs -> +1 if edges turn clockwise, -1 if anticlockwise
    # used later to discern bevel positions
    miter_signs = _sign_cross(full_normals[1:], full_normals[:-1])
    miters = 0.5 * miters

    # generate centers/offsets
    centers = np.repeat(path, 2, axis=0)
    offsets = np.repeat(miters, 2, axis=0)
    offsets[::2] *= -1

    triangles0 = np.tile(np.array([[0, 1, 3], [0, 3, 2]]), (len(path) - 1, 1))
    triangles = triangles0 + 2 * np.repeat(
        np.arange(len(path) - 1)[:, np.newaxis], 2, 0
    )

    # get vertex indices that are to be beveled
    idx_bevel = np.where(
        np.bitwise_or(bevel, miter_lengths_squared > (limit**2))
    )[0]

    if len(idx_bevel) > 0:
        idx_offset = (miter_signs[idx_bevel] < 0).astype(int)

        # outside and inside offsets are treated differently (only outside offsets get beveled)
        # See drawing at:
        # https://github.com/napari/napari/pull/6706#discussion_r1528790407
        idx_bevel_outside = 2 * idx_bevel + idx_offset
        idx_bevel_inside = 2 * idx_bevel + (1 - idx_offset)
        sign_bevel = np.expand_dims(miter_signs[idx_bevel], -1)

        # adjust offset of outer offset
        offsets[idx_bevel_outside] = (
            -0.5 * full_normals[:-1][idx_bevel] * sign_bevel
        )
        # adjust/normalize length of inner offset
        offsets[idx_bevel_inside] /= np.sqrt(
            miter_lengths_squared[idx_bevel, np.newaxis]
        )

        # special cases for the last vertex
        _nonspecial = idx_bevel != len(path) - 1

        idx_bevel = idx_bevel[_nonspecial]
        idx_bevel_outside = idx_bevel_outside[_nonspecial]
        sign_bevel = sign_bevel[_nonspecial]
        idx_offset = idx_offset[_nonspecial]

        # create new "right" bevel vertices to be added later
        centers_bevel = path[idx_bevel]
        offsets_bevel = -0.5 * full_normals[1:][idx_bevel] * sign_bevel

        n_centers = len(centers)
        # change vertices of triangles to the newly added right vertices
        triangles[2 * idx_bevel, idx_offset] = len(centers) + np.arange(
            len(idx_bevel)
        )
        triangles[2 * idx_bevel + (1 - idx_offset), idx_offset] = (
            n_centers + np.arange(len(idx_bevel))
        )

        # add a new center/bevel triangle
        triangles0 = np.tile(np.array([[0, 1, 2]]), (len(idx_bevel), 1))
        triangles_bevel = np.array(
            [
                2 * idx_bevel + idx_offset,
                2 * idx_bevel + (1 - idx_offset),
                n_centers + np.arange(len(idx_bevel)),
            ]
        ).T
        # add all new centers, offsets, and triangles
        centers = np.concatenate([centers, centers_bevel])
        offsets = np.concatenate([offsets, offsets_bevel])
        triangles = np.concatenate([triangles, triangles_bevel])

    # extracting vectors (~4x faster than np.moveaxis)
    a, b, c = tuple((centers + offsets)[triangles][:, i] for i in range(3))
    # flip negative oriented triangles
    flip_idx = _sign_cross(b - a, c - a) < 0
    triangles[flip_idx] = np.flip(triangles[flip_idx], axis=-1)

    return centers, offsets, triangles


def create_box_from_bounding_py(bounding_box: npt.NDArray) -> npt.NDArray:
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
    tl = bounding_box[(0, 0), (0, 1)]
    br = bounding_box[(1, 1), (0, 1)]
    tr = bounding_box[(1, 0), (0, 1)]
    bl = bounding_box[(0, 1), (0, 1)]
    return np.array(
        [
            tl,
            (tl + tr) / 2,
            tr,
            (tr + br) / 2,
            br,
            (br + bl) / 2,
            bl,
            (bl + tl) / 2,
            (tl + br) / 2,
        ]
    )


CACHE_WARMUP = False
USE_COMPILED_BACKEND = False

try:
    from PartSegCore_compiled_backend.triangulate import (
        triangulate_path_edge_numpy,
    )

except ImportError:
    triangulate_path_edge_numpy = None

try:
    from napari.layers.shapes._accelerated_triangulate import (
        create_box_from_bounding,
        generate_2D_edge_meshes,
        remove_path_duplicates,
    )

    def warmup_numba_cache() -> None:
        global CACHE_WARMUP
        if CACHE_WARMUP:
            return

        CACHE_WARMUP = True
        for order in ('C', 'F'):
            data = np.array(
                [[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.float32, order=order
            )
            data2 = np.array([[1, 1], [10, 15]], dtype=np.float32, order=order)

            if not (
                USE_COMPILED_BACKEND
                and triangulate_path_edge_numpy is not None
            ):
                generate_2D_edge_meshes(data, True)
                generate_2D_edge_meshes(data, False)
            remove_path_duplicates(data, False)
            remove_path_duplicates(data, True)
            create_box_from_bounding(data2)

except ImportError:
    generate_2D_edge_meshes = generate_2D_edge_meshes_py
    remove_path_duplicates = remove_path_duplicates_np
    create_box_from_bounding = create_box_from_bounding_py

    def warmup_numba_cache() -> None:
        # no numba, nothing to warm up
        pass
