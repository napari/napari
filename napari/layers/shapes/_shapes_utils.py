from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
from skimage.draw import line, polygon2mask
from vispy.geometry import PolygonData
from vispy.visuals.tube import _frenet_frames

from napari.layers.utils.layer_utils import segment_normal
from napari.utils.translations import trans

if TYPE_CHECKING:
    import numpy.typing as npt

try:
    # see https://github.com/vispy/vispy/issues/1029
    from triangle import triangulate
except ModuleNotFoundError:
    triangulate = None


def inside_boxes(boxes):
    """Checks which boxes contain the origin. Boxes need not be axis aligned

    Parameters
    ----------
    boxes : (N, 8, 2) array
        Array of N boxes that should be checked

    Returns
    -------
    inside : (N,) array of bool
        True if corresponding box contains the origin.
    """

    AB = boxes[:, 0] - boxes[:, 6]
    AM = boxes[:, 0]
    BC = boxes[:, 6] - boxes[:, 4]
    BM = boxes[:, 6]

    ABAM = np.multiply(AB, AM).sum(1)
    ABAB = np.multiply(AB, AB).sum(1)
    BCBM = np.multiply(BC, BM).sum(1)
    BCBC = np.multiply(BC, BC).sum(1)

    c1 = 0 <= ABAM
    c2 = ABAM <= ABAB
    c3 = 0 <= BCBM
    c4 = BCBM <= BCBC

    inside = np.all(np.array([c1, c2, c3, c4]), axis=0)

    return inside


def triangles_intersect_box(triangles, corners):
    """Determines which triangles intersect an axis aligned box.

    Parameters
    ----------
    triangles : (N, 3, 2) array
        Array of vertices of triangles to be tested
    corners : (2, 2) array
        Array specifying corners of a box

    Returns
    -------
    intersects : (N,) array of bool
        Array with `True` values for triangles intersecting the box
    """

    vertices_inside = triangle_vertices_inside_box(triangles, corners)
    edge_intersects = triangle_edges_intersect_box(triangles, corners)

    intersects = np.logical_or(vertices_inside, edge_intersects)

    return intersects


def triangle_vertices_inside_box(triangles, corners):
    """Determines which triangles have vertices inside an axis aligned box.

    Parameters
    ----------
    triangles : (N, 3, 2) array
        Array of vertices of triangles to be tested
    corners : (2, 2) array
        Array specifying corners of a box

    Returns
    -------
    inside : (N,) array of bool
        Array with `True` values for triangles with vertices inside the box
    """
    box = create_box(corners)[[0, 4]]

    vertices_inside = np.empty(triangles.shape[:-1], dtype=bool)
    for i in range(3):
        # check if each triangle vertex is inside the box
        below_top = np.all(box[1] >= triangles[:, i, :], axis=1)
        above_bottom = np.all(triangles[:, i, :] >= box[0], axis=1)
        vertices_inside[:, i] = np.logical_and(below_top, above_bottom)

    inside = np.any(vertices_inside, axis=1)

    return inside


def triangle_edges_intersect_box(triangles, corners):
    """Determines which triangles have edges that intersect the edges of an
    axis aligned box.

    Parameters
    ----------
    triangles : (N, 3, 2) array
        Array of vertices of triangles to be tested
    corners : (2, 2) array
        Array specifying corners of a box

    Returns
    -------
    intersects : (N,) array of bool
        Array with `True` values for triangles with edges that intersect the
        edges of the box.
    """
    box = create_box(corners)[[0, 2, 4, 6]]

    intersects = np.zeros([len(triangles), 12], dtype=bool)
    for i in range(3):
        # check if each triangle edge
        p1 = triangles[:, i, :]
        q1 = triangles[:, (i + 1) % 3, :]

        for j in range(4):
            # Check the four edges of the box
            p2 = box[j]
            q2 = box[(j + 1) % 3]
            intersects[:, i * 3 + j] = [
                lines_intersect(p1[k], q1[k], p2, q2) for k in range(len(p1))
            ]

    return np.any(intersects, axis=1)


def lines_intersect(p1, q1, p2, q2):
    """Determines if line segment p1q1 intersects line segment p2q2

    Parameters
    ----------
    p1 : (2,) array
        Array of first point of first line segment
    q1 : (2,) array
        Array of second point of first line segment
    p2 : (2,) array
        Array of first point of second line segment
    q2 : (2,) array
        Array of second point of second line segment

    Returns
    -------
    intersects : bool
        Bool indicating if line segment p1q1 intersects line segment p2q2
    """
    # Determine four orientations
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # Test general case
    if (o1 != o2) and (o3 != o4):
        return True

    # Test special cases
    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    # p1, q1 and q2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    # Doesn't fall into any special cases
    return False


def on_segment(p, q, r):
    """Checks if q is on the segment from p to r

    Parameters
    ----------
    p : (2,) array
        Array of first point of segment
    q : (2,) array
        Array of point to check if on segment
    r : (2,) array
        Array of second point of segment

    Returns
    -------
    on : bool
        Bool indicating if q is on segment from p to r
    """
    if (
        q[0] <= max(p[0], r[0])
        and q[0] >= min(p[0], r[0])
        and q[1] <= max(p[1], r[1])
        and q[1] >= min(p[1], r[1])
    ):
        on = True
    else:
        on = False

    return on


def orientation(p, q, r):
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
        if counterclockwise.
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    val = np.sign(val)

    return val


def is_collinear(points):
    """Determines if a list of 2D points are collinear.

    Parameters
    ----------
    points : (N, 2) array
        Points to be tested for collinearity

    Returns
    -------
    val : bool
        True is all points are collinear, False otherwise.
    """
    if len(points) < 3:
        return True

    # The collinearity test takes three points, the first two are the first
    # two in the list, and then the third is iterated through in the loop
    return all(orientation(points[0], points[1], p) == 0 for p in points[2:])


def point_to_lines(point, lines):
    """Calculate the distance between a point and line segments and returns the
    index of the closest line. First calculates the distance to the infinite
    line, then checks if the projected point lies between the line segment
    endpoints. If not, calculates distance to the endpoints

    Parameters
    ----------
    point : np.ndarray
        1x2 array of specifying the point
    lines : np.ndarray
        Nx2x2 array of line segments

    Returns
    -------
    index : int
        Integer index of the closest line
    location : float
        Normalized location of intersection of the distance normal to the line
        closest. Less than 0 means an intersection before the line segment
        starts. Between 0 and 1 means an intersection inside the line segment.
        Greater than 1 means an intersection after the line segment ends
    """

    # shift and normalize vectors
    lines_vectors = lines[:, 1] - lines[:, 0]
    point_vectors = point - lines[:, 0]
    end_point_vectors = point - lines[:, 1]
    norm_lines = np.linalg.norm(lines_vectors, axis=1, keepdims=True)
    reject = (norm_lines == 0).squeeze()
    norm_lines[reject] = 1
    unit_lines = lines_vectors / norm_lines

    # calculate distance to line
    line_dist = abs(np.cross(unit_lines, point_vectors))

    # calculate scale
    line_loc = (unit_lines * point_vectors).sum(axis=1) / norm_lines.squeeze()

    # for points not falling inside segment calculate distance to appropriate
    # endpoint
    line_dist[line_loc < 0] = np.linalg.norm(
        point_vectors[line_loc < 0], axis=1
    )
    line_dist[line_loc > 1] = np.linalg.norm(
        end_point_vectors[line_loc > 1], axis=1
    )
    line_dist[reject] = np.linalg.norm(point_vectors[reject], axis=1)
    line_loc[reject] = 0.5

    # calculate closet line
    index = np.argmin(line_dist)
    location = line_loc[index]

    return index, location


def create_box(data):
    """Creates the axis aligned interaction box of a list of points

    Parameters
    ----------
    data : np.ndarray
        Nx2 array of points whose interaction box is to be found

    Returns
    -------
    box : np.ndarray
        9x2 array of vertices of the interaction box. The first 8 points are
        the corners and midpoints of the box in clockwise order starting in the
        upper-left corner. The last point is the center of the box
    """
    min_val = [data[:, 0].min(axis=0), data[:, 1].min(axis=0)]
    max_val = [data[:, 0].max(axis=0), data[:, 1].max(axis=0)]
    tl = np.array([min_val[0], min_val[1]])
    tr = np.array([max_val[0], min_val[1]])
    br = np.array([max_val[0], max_val[1]])
    bl = np.array([min_val[0], max_val[1]])
    box = np.array(
        [
            tl,
            (tl + tr) / 2,
            tr,
            (tr + br) / 2,
            br,
            (br + bl) / 2,
            bl,
            (bl + tl) / 2,
            (tl + tr + br + bl) / 4,
        ]
    )
    return box


def rectangle_to_box(data):
    """Converts the four corners of a rectangle into a interaction box like
    representation. If the rectangle is not axis aligned the resulting box
    representation will not be axis aligned either

    Parameters
    ----------
    data : np.ndarray
        4xD array of corner points to be converted to a box like representation

    Returns
    -------
    box : np.ndarray
        9xD array of vertices of the interaction box. The first 8 points are
        the corners and midpoints of the box in clockwise order starting in the
        upper-left corner. The last point is the center of the box
    """
    if not data.shape[0] == 4:
        raise ValueError(
            trans._(
                "Data shape does not match expected `[4, D]` shape specifying corners for the rectangle",
                deferred=True,
            )
        )
    box = np.array(
        [
            data[0],
            (data[0] + data[1]) / 2,
            data[1],
            (data[1] + data[2]) / 2,
            data[2],
            (data[2] + data[3]) / 2,
            data[3],
            (data[3] + data[0]) / 2,
            data.mean(axis=0),
        ]
    )
    return box


def find_corners(data):
    """Finds the four corners of the interaction box defined by an array of
    points

    Parameters
    ----------
    data : np.ndarray
        Nx2 array of points whose interaction box is to be found

    Returns
    -------
    corners : np.ndarray
        4x2 array of corners of the bounding box
    """
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)
    tl = np.array([min_val[0], min_val[1]])
    tr = np.array([max_val[0], min_val[1]])
    br = np.array([max_val[0], max_val[1]])
    bl = np.array([min_val[0], max_val[1]])
    corners = np.array([tl, tr, br, bl])
    return corners


def center_radii_to_corners(center, radii):
    """Expands a center and radii into a four corner rectangle

    Parameters
    ----------
    center : np.ndarray | list
        Length 2 array or list of the center coordinates
    radii : np.ndarray | list
        Length 2 array or list of the two radii

    Returns
    -------
    corners : np.ndarray
        4x2 array of corners of the bounding box
    """
    data = np.array([center + radii, center - radii])
    corners = find_corners(data)
    return corners


def triangulate_ellipse(corners, num_segments=100):
    """Determines the triangulation of a path. The resulting `offsets` can
    multiplied by a `width` scalar and be added to the resulting `centers`
    to generate the vertices of the triangles for the triangulation, i.e.
    `vertices = centers + width*offsets`. Using the `centers` and `offsets`
    representation thus allows for the computed triangulation to be
    independent of the line width.

    Parameters
    ----------
    corners : np.ndarray
        4xD array of four bounding corners of the ellipse. The ellipse will
        still be computed properly even if the rectangle determined by the
        corners is not axis aligned. D in {2,3}
    num_segments : int
        Integer determining the number of segments to use when triangulating
        the ellipse

    Returns
    -------
    vertices : np.ndarray
        Mx2/Mx3 array coordinates of vertices for triangulating an ellipse.
        Includes the center vertex of the ellipse, followed by `num_segments`
        vertices around the boundary of the ellipse (M = `num_segments`+1)
    triangles : np.ndarray
        Px3 array of the indices of the vertices for the triangles of the
        triangulation. Has length (P) given by `num_segments`,
        (P = M-1 = num_segments)

    Notes
    -----
    Despite it's name the ellipse will have num_segments-1 segments on their outline.
    That is to say num_segments=7 will lead to ellipses looking like hexagons.

    The behavior of this function is not well defined if the ellipse is degenerate
    in the current plane/volume you are currently observing.


    """
    if not corners.shape[0] == 4:
        raise ValueError(
            trans._(
                "Data shape does not match expected `[4, D]` shape specifying corners for the ellipse",
                deferred=True,
            )
        )
    assert corners.shape in {(4, 2), (4, 3)}
    center = corners.mean(axis=0)
    adjusted = corners - center

    # Take to consecutive corners difference
    # that give us the 1/2 minor and major axes.
    ax1 = (adjusted[1] - adjusted[0]) / 2
    ax2 = (adjusted[2] - adjusted[1]) / 2
    # Compute the transformation matrix from the unit circle
    # to our current ellipse.
    # ... it's easy just the 1/2 minor/major axes for the two column
    # note that our transform shape will depends on wether we are 2D-> 2D (matrix, 2 by 2),
    # or 2D -> 3D (matrix 2 by 3).
    transform = np.stack((ax1, ax2))
    if corners.shape == (4, 2):
        assert transform.shape == (2, 2)
    else:
        assert transform.shape == (2, 3)

    # we discretize the unit circle always in 2D.
    v2d = np.zeros((num_segments + 1, 2), dtype=np.float32)
    theta = np.linspace(0, np.deg2rad(360), num_segments)
    v2d[1:, 0] = np.cos(theta)
    v2d[1:, 1] = np.sin(theta)

    # ! vertices shape can be 2,M or 3,M depending on the transform.
    vertices = np.matmul(v2d, transform)

    # Shift back to center
    vertices = vertices + center

    triangles = (
        np.arange(num_segments) + np.array([[0], [1], [2]])
    ).T * np.array([0, 1, 1])
    triangles[-1, 2] = 1

    return vertices, triangles


def triangulate_face(data):
    """Determines the triangulation of the face of a shape.

    Parameters
    ----------
    data : np.ndarray
        Nx2 array of vertices of shape to be triangulated

    Returns
    -------
    vertices : np.ndarray
        Mx2 array vertices of the triangles.
    triangles : np.ndarray
        Px3 array of the indices of the vertices that will form the
        triangles of the triangulation
    """

    if triangulate is not None:
        len_data = len(data)

        edges = np.empty((len_data, 2), dtype=np.uint32)
        edges[:, 0] = np.arange(len_data)
        edges[:, 1] = np.arange(1, len_data + 1)
        # connect last with first vertex
        edges[-1, 1] = 0

        res = triangulate({"vertices": data, "segments": edges}, "p")
        vertices, triangles = res['vertices'], res['triangles']
    else:
        vertices, triangles = PolygonData(vertices=data).triangulate()

    triangles = triangles.astype(np.uint32)

    return vertices, triangles


def triangulate_edge(path, closed=False):
    """Determines the triangulation of a path. The resulting `offsets` can
    multiplied by a `width` scalar and be added to the resulting `centers`
    to generate the vertices of the triangles for the triangulation, i.e.
    `vertices = centers + width*offsets`. Using the `centers` and `offsets`
    representation thus allows for the computed triangulation to be
    independent of the line width.

    Parameters
    ----------
    path : np.ndarray
        Nx2 or Nx3 array of central coordinates of path to be triangulated
    closed : bool
        Bool which determines if the path is closed or not.

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

    path = np.asanyarray(path)

    # Remove any equal adjacent points
    if len(path) > 2:
        idx = np.concatenate([[True], ~np.all(path[1:] == path[:-1], axis=-1)])
        clean_path = path[idx].copy()

        if clean_path.shape[0] == 1:
            clean_path = np.concatenate((clean_path, clean_path), axis=0)
    else:
        clean_path = path

    if clean_path.shape[-1] == 2:
        centers, offsets, triangles = generate_2D_edge_meshes(
            clean_path, closed=closed
        )
    else:
        centers, offsets, triangles = generate_tube_meshes(
            clean_path, closed=closed
        )

    return centers, offsets, triangles


def _mirror_point(x, y):
    return 2 * y - x


def _sign_nonzero(x):
    y = np.sign(x).astype(int)
    y[y == 0] = 1
    return y


def _sign_cross(x, y):
    """sign of cross product (faster for 2d)"""
    if x.shape[1] == y.shape[1] == 2:
        return _sign_nonzero(x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0])
    if x.shape[1] == y.shape[1] == 3:
        return _sign_nonzero(np.cross(x, y))

    raise ValueError(x.shape[1], y.shape[1])


def generate_2D_edge_meshes(path, closed=False, limit=3, bevel=False):
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

    # extend path by adding a vertex at beginning and end
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
        # only the 'outwards sticking' offsets should be changed
        # TODO: This is not entirely true as in extreme cases both can go to infinity
        idx_offset = (miter_signs[idx_bevel] < 0).astype(int)
        idx_bevel_full = 2 * idx_bevel + idx_offset
        sign_bevel = np.expand_dims(miter_signs[idx_bevel], -1)

        # adjust offset of outer "left" vertex
        offsets[idx_bevel_full] = (
            -0.5 * full_normals[:-1][idx_bevel] * sign_bevel
        )

        # special cases for the last vertex
        _nonspecial = idx_bevel != len(path) - 1

        idx_bevel = idx_bevel[_nonspecial]
        idx_bevel_full = idx_bevel_full[_nonspecial]
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
        triangles[
            2 * idx_bevel + (1 - idx_offset), idx_offset
        ] = n_centers + np.arange(len(idx_bevel))

        # add center triangle
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


def generate_tube_meshes(path, closed=False, tube_points=10):
    """Generates list of mesh vertices and triangles from a path

    Adapted from vispy.visuals.TubeVisual
    https://github.com/vispy/vispy/blob/main/vispy/visuals/tube.py

    Parameters
    ----------
    path : (N, 3) array
        Vertices specifying the path.
    closed : bool
        Bool which determines if the path is closed or not.
    tube_points : int
        The number of points in the circle-approximating polygon of the
        tube's cross section.

    Returns
    -------
    centers : (M, 3) array
        Vertices of all triangles for the lines
    offsets : (M, D) array
        offsets of all triangles for the lines
    triangles : (P, 3) array
        Vertex indices that form the mesh triangles
    """
    points = np.array(path).astype(float)

    if closed and not np.all(points[0] == points[-1]):
        points = np.concatenate([points, [points[0]]], axis=0)

    tangents, normals, binormals = _frenet_frames(points, closed)

    segments = len(points) - 1

    # get the positions of each vertex
    grid = np.zeros((len(points), tube_points, 3))
    grid_off = np.zeros((len(points), tube_points, 3))
    for i in range(len(points)):
        pos = points[i]
        normal = normals[i]
        binormal = binormals[i]

        # Add a vertex for each point on the circle
        v = np.arange(tube_points, dtype=float) / tube_points * 2 * np.pi
        cx = -1.0 * np.cos(v)
        cy = np.sin(v)
        grid[i] = pos
        grid_off[i] = cx[:, np.newaxis] * normal + cy[:, np.newaxis] * binormal

    # construct the mesh
    indices = []
    for i in range(segments):
        for j in range(tube_points):
            ip = (i + 1) % segments if closed else i + 1
            jp = (j + 1) % tube_points

            index_a = i * tube_points + j
            index_b = ip * tube_points + j
            index_c = ip * tube_points + jp
            index_d = i * tube_points + jp

            indices.append([index_a, index_b, index_d])
            indices.append([index_b, index_c, index_d])
    triangles = np.array(indices, dtype=np.uint32)

    centers = grid.reshape(grid.shape[0] * grid.shape[1], 3)
    offsets = grid_off.reshape(grid_off.shape[0] * grid_off.shape[1], 3)

    return centers, offsets, triangles


def path_to_mask(mask_shape, vertices):
    """Converts a path to a boolean mask with `True` for points lying along
    each edge.

    Parameters
    ----------
    mask_shape : array (2,)
        Shape of mask to be generated.
    vertices : array (N, 2)
        Vertices of the path.

    Returns
    -------
    mask : np.ndarray
        Boolean array with `True` for points along the path

    """
    mask_shape = np.asarray(mask_shape, dtype=int)
    mask = np.zeros(mask_shape, dtype=bool)

    vertices = np.round(np.clip(vertices, 0, mask_shape - 1)).astype(int)

    # remove identical, consecutive vertices
    duplicates = np.all(np.diff(vertices, axis=0) == 0, axis=-1)
    duplicates = np.concatenate(([False], duplicates))
    vertices = vertices[~duplicates]

    iis, jjs = [], []
    for v1, v2 in zip(vertices, vertices[1:]):
        ii, jj = line(*v1, *v2)
        iis.extend(ii.tolist())
        jjs.extend(jj.tolist())

    mask[iis, jjs] = 1

    return mask


def poly_to_mask(mask_shape, vertices):
    """Converts a polygon to a boolean mask with `True` for points
    lying inside the shape. Uses the bounding box of the vertices to reduce
    computation time.

    Parameters
    ----------
    mask_shape : np.ndarray | tuple
        1x2 array of shape of mask to be generated.
    vertices : np.ndarray
        Nx2 array of the vertices of the polygon.

    Returns
    -------
    mask : np.ndarray
        Boolean array with `True` for points inside the polygon
    """
    return polygon2mask(mask_shape, vertices)


def grid_points_in_poly(shape, vertices):
    """Converts a polygon to a boolean mask with `True` for points
    lying inside the shape. Loops through all indices in the grid

    Parameters
    ----------
    shape : np.ndarray | tuple
        1x2 array of shape of mask to be generated.
    vertices : np.ndarray
        Nx2 array of the vertices of the polygon.

    Returns
    -------
    mask : np.ndarray
        Boolean array with `True` for points inside the polygon
    """
    points = np.array(
        [(x, y) for x in range(shape[0]) for y in range(shape[1])], dtype=int
    )
    inside = points_in_poly(points, vertices)
    mask = inside.reshape(shape)
    return mask


def points_in_poly(points, vertices):
    """Tests points for being inside a polygon using the ray casting algorithm

    Parameters
    ----------
    points : np.ndarray
        Mx2 array of points to be tested
    vertices : np.ndarray
        Nx2 array of the vertices of the polygon.

    Returns
    -------
    inside : np.ndarray
        Length M boolean array with `True` for points inside the polygon
    """
    n_verts = len(vertices)
    inside = np.zeros(len(points), dtype=bool)
    j = n_verts - 1
    for i in range(n_verts):
        # Determine if a horizontal ray emanating from the point crosses the
        # line defined by vertices i-1 and vertices i.
        cond_1 = np.logical_and(
            vertices[i, 1] <= points[:, 1], points[:, 1] < vertices[j, 1]
        )
        cond_2 = np.logical_and(
            vertices[j, 1] <= points[:, 1], points[:, 1] < vertices[i, 1]
        )
        cond_3 = np.logical_or(cond_1, cond_2)
        d = vertices[j] - vertices[i]
        # Prevents floating point imprecision from generating false positives
        tolerance = 1e-12
        d = np.where(abs(d) < tolerance, 0, d)
        if d[1] == 0:
            # If y vertices are aligned avoid division by zero
            cond_4 = d[0] * (points[:, 1] - vertices[i, 1]) > 0
        else:
            cond_4 = points[:, 0] < (
                d[0] * (points[:, 1] - vertices[i, 1]) / d[1] + vertices[i, 0]
            )
        cond_5 = np.logical_and(cond_3, cond_4)
        inside[cond_5] = 1 - inside[cond_5]
        j = i

    # If the number of crossings is even then the point is outside the polygon,
    # if the number of crossings is odd then the point is inside the polygon

    return inside


def extract_shape_type(data, shape_type=None):
    """Separates shape_type from data if present, and returns both.

    Parameters
    ----------
    data : Array | Tuple(Array,str) | List[Array | Tuple(Array, str)] | Tuple(List[Array], str)
        list or array of vertices belonging to each shape, optionally containing shape type strings
    shape_type : str | None
        metadata shape type string, or None if none was passed

    Returns
    -------
    data : Array | List[Array]
        list or array of vertices belonging to each shape
    shape_type : List[str] | None
        type of each shape in data, or None if none was passed
    """
    # Tuple for one shape or list of shapes with shape_type
    if isinstance(data, Tuple):
        shape_type = data[1]
        data = data[0]
    # List of (vertices, shape_type) tuples
    elif len(data) != 0 and all(isinstance(datum, Tuple) for datum in data):
        shape_type = [datum[1] for datum in data]
        data = [datum[0] for datum in data]
    return data, shape_type


def get_default_shape_type(current_type):
    """If all shapes in current_type are of identical shape type,
       return this shape type, else "polygon" as lowest common
       denominator type.

    Parameters
    ----------
    current_type : list of str
        list of current shape types

    Returns
    -------
    default_type : str
        default shape type
    """
    default = "polygon"
    if not current_type:
        return default
    first_type = current_type[0]
    if all(shape_type == first_type for shape_type in current_type):
        return first_type
    return default


def get_shape_ndim(data):
    """Checks whether data is a list of the same type of shape, one shape, or
    a list of different shapes and returns the dimensionality of the shape/s.

    Parameters
    ----------
    data : (N, ) list of array
        List of shape data, where each element is an (N, D) array of the
        N vertices of a shape in D dimensions.

    Returns
    -------
    ndim : int
        Dimensionality of the shape/s in data
    """
    # list of all the same shapes
    if np.array(data, dtype=object).ndim == 3:
        ndim = np.array(data).shape[2]
    # just one shape
    elif np.array(data[0]).ndim == 1:
        ndim = np.array(data).shape[1]
    # list of different shapes
    else:
        ndim = np.array(data[0]).shape[1]
    return ndim


def number_of_shapes(data):
    """Determine number of shapes in the data.

    Parameters
    ----------
    data : list or np.ndarray
        Can either be no shapes, if empty, a
        single shape or a list of shapes.

    Returns
    -------
    n_shapes : int
        Number of new shapes
    """
    if len(data) == 0:
        # If no new shapes
        n_shapes = 0
    elif np.array(data[0]).ndim == 1:
        # If a single array for a shape
        n_shapes = 1
    else:
        n_shapes = len(data)

    return n_shapes


def validate_num_vertices(
    data, shape_type, min_vertices=None, valid_vertices=None
):
    """Raises error if a shape in data has invalid number of vertices.

    Checks whether all shapes in data have a valid number of vertices
    for the given shape type and vertex information. Rectangles and
    ellipses can have either 2 or 4 vertices per shape,
    lines can have only 2, while paths and polygons have a minimum
    number of vertices, but no maximum.

    One of valid_vertices or min_vertices must be passed to the
    function.

    Parameters
    ----------
    data : Array | Tuple(Array,str) | List[Array | Tuple(Array, str)] | Tuple(List[Array], str)
        List of shape data, where each element is either an (N, D) array of the
        N vertices of a shape in D dimensions or a tuple containing an array of
        the N vertices and the shape_type string. Can be an 3-dimensional array
        if each shape has the same number of vertices.
    shape_type : str
        Type of shape being validated (for detailed error message)
    min_vertices : int or None
        Minimum number of vertices for the shape type, by default None
    valid_vertices : Tuple(int) or None
        Valid number of vertices for the shape type in data, by default None

    Raises
    ------
    ValueError
        Raised if a shape is found with invalid number of vertices
    """
    n_shapes = number_of_shapes(data)
    # single array of vertices
    if n_shapes == 1 and len(np.array(data).shape) == 2:
        # wrap in extra dimension so we can iterate through shape not vertices
        data = [data]
    for shape in data:
        if (valid_vertices and len(shape) not in valid_vertices) or (
            min_vertices and len(shape) < min_vertices
        ):
            raise ValueError(
                trans._(
                    "{shape_type} {shape} has invalid number of vertices: {shape_length}.",
                    deferred=True,
                    shape_type=shape_type,
                    shape=shape,
                    shape_length=len(shape),
                )
            )


def perpendicular_distance(
    point: npt.NDArray, line_start: npt.NDArray, line_end: npt.NDArray
) -> float:
    """Calculate the perpendicular distance of a point to a given euclidean line.

    Calculates the shortest distance of a point to a euclidean line defined by a line_start point and a line_end point.
    Works up to any dimension.

    Parameters
    ---------
    point : np.ndarray
        A point defined by a numpy array of shape (viewer.ndims,)  which is part of a polygon shape.
    line_start : np.ndarray
        A point defined by a numpy array of shape (viewer.ndims,)  used to define the starting point of a line.
    line_end : np.ndarray
        A point defined by a numpy array of shape (viewer.ndims,)  used to define the end point of a line.

    Returns
    -------
    float
        A float number representing the distance of point to a euclidean line defined by line_start and line_end.
    """

    if np.array_equal(line_start, line_end):
        return np.linalg.norm(point - line_start)

    t = np.dot(point - line_end, line_start - line_end) / np.dot(
        line_start - line_end, line_start - line_end
    )
    return np.linalg.norm(t * (line_start - line_end) + line_end - point)


def rdp(vertices: npt.NDArray, epsilon: float) -> npt.NDArray:
    """Reduce the number of vertices that make up a polygon.

    Implementation of the Ramer-Douglas-Peucker algorithm based on:
    https://github.com/fhirschmann/rdp/blob/master/rdp. This algorithm reduces the amounts of points in a polyline or
    in this case reduces the number of vertices in a polygon shape.

    Parameters
    ----------
    vertices : np.ndarray
        A numpy array of shape (n, viewer.ndims) containing the vertices of a polygon shape.
    epsilon : float
        A float representing the maximum distance threshold. When the perpendicular distance of a point to a given line
        is higher, subsequent refinement occurs.

    Returns
    -------
    np.ndarray
        A numpy array of shape (n, viewer.ndims) containing the vertices of a polygon shape.
    """
    max_distance_index = -1
    max_distance = 0.0

    for i in range(1, vertices.shape[0]):
        d = perpendicular_distance(vertices[i], vertices[0], vertices[-1])
        if d > max_distance:
            max_distance_index = i
            max_distance = d

    if epsilon != 0:
        if max_distance > epsilon and epsilon:
            l1 = rdp(vertices[: max_distance_index + 1], epsilon)
            l2 = rdp(vertices[max_distance_index:], epsilon)
            return np.vstack((l1[:-1], l2))

        # This part of the algorithm is actually responsible for removing the datapoints.
        return np.vstack((vertices[0], vertices[-1]))

    # When epsilon is 0, avoid removing datapoints
    return vertices
