from collections import defaultdict
from typing import Literal, overload

import numpy as np
import numpy.typing as npt

from napari.layers.shapes.shape_types import (
    CoordinateArray,
    CoordinateArray2D,
    CoordinateArray3D,
    EdgeArray,
)
from napari.layers.utils.layer_utils import segment_normal


def remove_path_duplicates_py(
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


@overload
def reconstruct_polygons_from_edges_py(
    vertices: CoordinateArray2D, edges: EdgeArray
) -> list[CoordinateArray2D]: ...


@overload
def reconstruct_polygons_from_edges_py(
    vertices: CoordinateArray3D, edges: EdgeArray
) -> list[CoordinateArray3D]: ...


def reconstruct_polygons_from_edges_py(
    vertices: CoordinateArray, edges: EdgeArray
) -> list[CoordinateArray2D] | list[CoordinateArray3D]:
    """
    Reconstruct polygons from vertices and edges.

    Parameters
    ----------
    vertices : np.ndarray
        Array of vertex coordinates with shape (N, 2) or (N, 3)
    edges : np.ndarray
        Array of edge indices with shape (M, 2)

    Returns
    -------
    list of np.ndarray
        List of polygons, where each polygon is an array of vertex coordinates
    """
    # Create an adjacency list representation from the edges
    adjacency = defaultdict(list)
    for edge_ in edges:
        v1, v2 = edge_
        adjacency[v1].append(v2)
        adjacency[v2].append(v1)

    # Initialize set of unvisited edges
    unvisited_edges: set[tuple[np.int64, np.int64]] = {
        (edge[0], edge[1]) for edge in edges
    }
    unvisited_edges.update({(edge[1], edge[0]) for edge in edges})

    # List to store resulting polygons
    polygons = []

    # Process each edge until all are visited
    while unvisited_edges:
        # Start with any unvisited edge
        edge: tuple[np.int64, np.int64] = next(iter(unvisited_edges))
        current_vertex = edge[0]
        start_vertex = edge[1]

        # Start a new polygon
        polygon_indices = [start_vertex]

        # Remove the first edge
        unvisited_edges.discard((start_vertex, current_vertex))
        unvisited_edges.discard((current_vertex, start_vertex))

        # Follow the edges to form a polygon
        while current_vertex != polygon_indices[0]:
            polygon_indices.append(current_vertex)

            # Find the next unvisited edge
            next_vertex = None
            for neighbor in adjacency[current_vertex]:
                if (current_vertex, neighbor) in unvisited_edges:
                    next_vertex = neighbor
                    unvisited_edges.discard((current_vertex, next_vertex))
                    unvisited_edges.discard((next_vertex, current_vertex))
                    break

            # If no unvisited edge was found, we have an open polyline
            if next_vertex is None:
                break

            current_vertex = next_vertex

        # Convert indices to coordinates and add to the result
        polygon_vertices = vertices[polygon_indices]
        polygons.append(polygon_vertices)

    return polygons


def normalize_vertices_and_edges_py(
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
    if tuple(vertices[0]) == tuple(vertices[-1]):  # closed polygon
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
        vertex_t = vertex[0], vertex[1]
        current_idx = vertex_to_idx.setdefault(vertex_t, i)
        if current_idx == i:
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
        vertex_t = tuple(vertices[-1])
        idx = vertex_to_idx[vertex_t]
        edge = (0, idx)
        if edge in edges:
            edges.remove(edge)
        else:
            edges.add(edge)

    # remove the edge of length 0 that is added at the first run of the above loop
    # because of the initialization of prev_idx to 0
    edges.remove((0, 0))

    new_vertices_array = np.array(new_vertices, dtype=np.float32)
    edges_array = np.array(list(edges), dtype=np.int64)
    return new_vertices_array, edges_array


def _are_polar_angles_monotonic(poly: npt.NDArray, orientation_: int) -> bool:
    """Check whether a polygon with same oriented angles between successive edges is simple.

    A polygon is considered simple if its edges do not intersect themselves.
    This is determined by checking whether the angles between successive
    vertices, measured from the centroid, increase consistently around the
    polygon in a counterclockwise (or clockwise) direction. If the angles
    from one vertex to the next increase, the polygon is simple.

    Parameters
    ----------
    poly: numpy array of floats, shape (N, 2)
        polygon vertices, in order.
    orientation_: int
        The orientation of the polygon. A value of `1` indicates clockwise
        and `-1` indicates counterclockwise orientation.

    Returns
    -------
    bool:
        if all angles are increasing return True, otherwise False
    """
    if poly.shape[0] < 3:  # pragma: no cover
        return False  # Not enough vertices to form a polygon
    if orientation_ == 1:
        poly = poly[::-1]
    centroid = poly.mean(axis=0)
    angles = np.arctan2(poly[:, 1] - centroid[1], poly[:, 0] - centroid[0])
    # orig_angles = angles.copy()
    shifted_angles = angles - angles[0]
    shifted_angles[shifted_angles < 0] += 2 * np.pi
    # check if angles are increasing
    return bool(np.all(np.diff(shifted_angles) > 0))


def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Determines oritentation of ordered triplet (p, q, r)

    Parameters
    ----------
    p : (2,) array
        Array of first point of triplet
    q : (2,) array
        Array of second point of triplet
    r : (2,) array
        Array of third point of triplet

    Returns
    -------
    val : int
        One of (-1, 0, 1). 0 if p, q, r are collinear, 1 if clockwise, and -1
        if counterclockwise. (These definitions assume napari's default
        reference frame, in which the 0th axis is y pointing down, and the
        1st axis is x, pointing right.)
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    val = np.sign(val)

    return val


def _common_orientation(poly: npt.NDArray) -> int | None:
    """Check whether a polygon has the same orientation for all its angles.
    and return the orientation.

    Parameters
    ----------
    poly: numpy array of floats, shape (N, 3)
        Polygon vertices, in order.

    Returns
    -------
    int or None
        if all angles have same orientation return it, otherwise None.
        Possible values: -1 if all angles are counterclockwise, 0 if all angles
        are collinear, 1 if all angles are clockwise.(These definitions
        assume napari's default reference frame, in which the 0th axis is y,
        pointing down, and the 1st axis is x, pointing right.)
    """
    if poly.shape[0] < 3:
        return None
    fst = poly[:-2]
    snd = poly[1:-1]
    thrd = poly[2:]
    orn_set = np.unique(orientation(fst.T, snd.T, thrd.T))
    if orn_set.size != 1:
        return None
    if (orn_set[0] == orientation(poly[-2], poly[-1], poly[0])) and (
        orn_set[0] == orientation(poly[-1], poly[0], poly[1])
    ):
        return int(orn_set[0])
    return None


def is_convex_py(poly: npt.NDArray) -> bool:
    """Check whether a polygon is convex.

    Parameters
    ----------
    poly: numpy array of floats, shape (N, 3)
        Polygon vertices, in order.

    Returns
    -------
    bool
        True if the given polygon is convex.
    """
    orientation_ = _common_orientation(poly)
    if orientation_ is None or orientation_ == 0:
        return False
    return _are_polar_angles_monotonic(poly, orientation_)
