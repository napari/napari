from typing import Dict

import numpy as np


def project_point_to_plane(
    point: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray
) -> np.ndarray:
    """Project a point on to a plane that has
    been defined as a point and a normal vector.

    Parameters
    ----------
    point : np.ndarray
        The coordinate of the point to be projected.
        Should have shape (N,3).
    plane_point : np.ndarray
        The point on the plane used to define the plane.
        Should have shape (3,).
    plane_normal : np.ndarray
        The normal vector used to define the plane.
        Should be a unit vector and have shape (3,).

    Returns
    -------
    projected_point : np.ndarray
        The point that has been projected to the plane.
        This is always an Nx3 array.
    """
    point = np.asarray(point)
    if point.ndim == 1:
        point = np.expand_dims(point, axis=0)
    plane_point = np.asarray(plane_point)
    # make the plane normals have the same shape as the points
    plane_normal = np.tile(plane_normal, (point.shape[0], 1))

    # get the vector from point on the plane
    # to the point to be projected
    point_vector = point - plane_point

    # find the distance to the plane along the normal direction
    dist_to_plane = np.multiply(point_vector, plane_normal).sum(axis=1)

    # project the point
    projected_point = point - (dist_to_plane[:, np.newaxis] * plane_normal)

    return projected_point


def rotation_matrix_from_vectors(vec_1, vec_2):
    """Calculate the rotation matrix that aligns vec1 to vec2.

    Parameters
    ----------
    vec_1 : np.ndarray
        The vector you want to rotate
    vec_2 : np.ndarray
        The vector you would like to align to.
    Returns
    -------
    rotation_matrix : np.ndarray
        The rotation matrix that aligns vec_1 with vec_2.
        That is rotation_matrix.dot(vec_1) == vec_2
    """
    vec_1 = (vec_1 / np.linalg.norm(vec_1)).reshape(3)
    vec_2 = (vec_2 / np.linalg.norm(vec_2)).reshape(3)
    cross_prod = np.cross(vec_1, vec_2)
    dot_prod = np.dot(vec_1, vec_2)
    if any(cross_prod):  # if not all zeros then
        s = np.linalg.norm(cross_prod)
        kmat = np.array(
            [
                [0, -cross_prod[2], cross_prod[1]],
                [cross_prod[2], 0, -cross_prod[0]],
                [-cross_prod[1], cross_prod[0], 0],
            ]
        )
        rotation_matrix = (
            np.eye(3) + kmat + kmat.dot(kmat) * ((1 - dot_prod) / (s ** 2))
        )

    else:
        if np.allclose(dot_prod, 1):
            # if the vectors are already aligned, return the identity
            rotation_matrix = np.eye(3)
        else:
            # if the vectors are in opposite direction, rotate 180 degrees
            rotation_matrix = np.diag([-1, -1, 1])
    return rotation_matrix


def clamp_point_to_bounding_box(point: np.ndarray, bounding_box: np.ndarray):
    """Ensure that a point is inside of the bounding box. If the point has a
    coordinate outside of the bounding box, the value is clipped to the max
    extent of the bounding box.

    Parameters
    ----------
    point : np.ndarray
        n-dimensional point as an (n,) ndarray
    bounding_box: np.ndarray
        n-dimensional bounding box as a (n, 2) ndarray

    Returns
    -------
    clamped_point : np.ndarray
        `point` clamped to the limits of `bounding_box`
    """
    clamped_point = np.clip(point, bounding_box[:, 0], bounding_box[:, 1] - 1)
    return clamped_point


def face_coordinate_from_bounding_box(
    bounding_box: np.ndarray, face_normal: np.ndarray
) -> float:
    """Get the coordinate for a given face in an axis-aligned bounding box.
    For example, if the bounding box has extents [[0, 10], [0, 20], [0, 30]]
    (ordered zyx), then the face with normal [0, 1, 0] is described by
    y=20. Thus, the face_coordinate in this case is 20.

    Parameters
    ----------
    bounding_box: np.ndarray
        n-dimensional bounding box as a (n, 2) ndarray.
        Each row should contain the [min, max] extents for the
        axis.
    face_normal : np.ndarray
        normal vector of the face as an (n,)  ndarray

    Returns
    -------
    face_coordinate : float
        The value where the bounding box face specified by face_normal intersects
        the axis its normal is aligned with.
    """
    axis = np.argwhere(face_normal)
    if face_normal[axis] > 0:
        # face is pointing in the positive direction,
        # take the max extent
        face_coordinate = bounding_box[axis, 1]
    else:
        # face is pointing in the negative direction,
        # take the min extent
        face_coordinate = bounding_box[axis, 0]
    return face_coordinate


def intersect_line_with_axis_aligned_plane(
    plane_intercept: float,
    plane_normal: np.ndarray,
    line_start: np.ndarray,
    line_direction: np.ndarray,
) -> np.ndarray:
    """Find the intersection of a ray with an axis aligned plane

    Parameters
    ----------
    plane_intercept: float
        The coordinate on the axis the plane is normal that the plane intersects.
        For example, if the plane is described by y=42, plane_intercept is 42.
    plane_normal : np.ndarray
        normal vector of the plane as an (n,)  ndarray
    line_start : np.ndarray
        start point of the line as an (n,) ndarray
    line_direction : np.ndarray
        direction vector of the line as an (n,) ndarray

    Returns
    -------
    intersection_point : np.ndarray
        point where the line intersects the axis aligned plane
    """
    # find the axis the plane exists in
    plane_axis = np.squeeze(np.argwhere(plane_normal))

    # get the intersection coordinate
    t = (plane_intercept - line_start[plane_axis]) / line_direction[plane_axis]
    intersection_point = line_start + t * line_direction

    return intersection_point


def bounding_box_to_face_vertices(
    bounding_box: np.ndarray,
) -> Dict[str, np.ndarray]:
    """From a layer bounding box (N, 2), N=ndim, return a dictionary containing
    the vertices of each face of the bounding_box.

    Parameters
    ----------
    bounding_box : np.ndarray
        (N, 2), N=ndim array with the min and max value for each dimension of
        the bounding box. The bounding box is take form the last
        three rows, which are assumed to be in order (z, y, x).

    Returns
    -------
    face_coords : Dict[str, np.ndarray]
        A dictionary containing the coordinates for the vertices for each face.
        The keys are strings: 'x_pos', 'x_neg', 'y_pos', 'y_neg', 'z_pos', 'z_neg'.
        'x_pos' is the face with the normal in the positive x direction and
        'x_neg' is the face with the normal in the negative direction.
        Coordinates are ordered (z, y, x).
    """
    x_min, x_max = bounding_box[-1, :]
    y_min, y_max = bounding_box[-2, :]
    z_min, z_max = bounding_box[-3, :]

    face_coords = {
        "x_pos": np.array(
            [
                [z_min, y_min, x_max],
                [z_min, y_max, x_max],
                [z_max, y_max, x_max],
                [z_max, y_min, x_max],
            ]
        ),
        "x_neg": np.array(
            [
                [z_min, y_min, x_min],
                [z_min, y_max, x_min],
                [z_max, y_max, x_min],
                [z_max, y_min, x_min],
            ]
        ),
        "y_pos": np.array(
            [
                [z_min, y_max, x_min],
                [z_min, y_max, x_max],
                [z_max, y_max, x_max],
                [z_max, y_max, x_min],
            ]
        ),
        "y_neg": np.array(
            [
                [z_min, y_min, x_min],
                [z_min, y_min, x_max],
                [z_max, y_min, x_max],
                [z_max, y_min, x_min],
            ]
        ),
        "z_pos": np.array(
            [
                [z_max, y_min, x_min],
                [z_max, y_min, x_max],
                [z_max, y_max, x_max],
                [z_max, y_max, x_min],
            ]
        ),
        "z_neg": np.array(
            [
                [z_min, y_min, x_min],
                [z_min, y_min, x_max],
                [z_min, y_max, x_max],
                [z_min, y_max, x_min],
            ]
        ),
    }
    return face_coords


def inside_triangles(triangles):
    """Checks which triangles contain the origin

    Parameters
    ----------
    triangles : (N, 3, 2) array
        Array of N triangles that should be checked

    Returns
    -------
    inside : (N,) array of bool
        Array with `True` values for trinagles containing the origin
    """

    AB = triangles[:, 1, :] - triangles[:, 0, :]
    AC = triangles[:, 2, :] - triangles[:, 0, :]
    BC = triangles[:, 2, :] - triangles[:, 1, :]

    s_AB = -AB[:, 0] * triangles[:, 0, 1] + AB[:, 1] * triangles[:, 0, 0] >= 0
    s_AC = -AC[:, 0] * triangles[:, 0, 1] + AC[:, 1] * triangles[:, 0, 0] >= 0
    s_BC = -BC[:, 0] * triangles[:, 1, 1] + BC[:, 1] * triangles[:, 1, 0] >= 0

    inside = np.all(np.array([s_AB != s_AC, s_AB == s_BC]), axis=0)

    return inside


def intersect_line_with_plane_3d(
    line_position: np.ndarray,
    line_direction: np.ndarray,
    plane_position: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    """
    Find the intersection of a line with an arbitrarily oriented plane in 3D.
    The line is defined by a position and a direction vector.
    The plane is defined by a position and a normal vector.
    https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection


    Parameters
    ----------
    line_position : np.ndarray
        a position on a 3D line with shape (3,).
    line_direction : np.ndarray
        direction of the 3D line with shape (3,).
    plane_position : np.ndarray
        a position on a plane in 3D with shape (3,).
    plane_normal : np.ndarray
        a vector normal to the plane in 3D with shape (3,).

    Returns
    -------
    plane_intersection : np.ndarray
        the intersection of the line with the plane, shape (3,)
    """
    # cast to arrays
    line_position = np.asarray(line_position, dtype=float)
    line_direction = np.asarray(line_direction, dtype=float)
    plane_position = np.asarray(plane_position, dtype=float)
    plane_normal = np.asarray(plane_normal, dtype=float)

    # project direction between line and plane onto the plane normal
    line_plane_direction = plane_position - line_position
    line_plane_on_plane_normal = np.dot(line_plane_direction, plane_normal)

    # project line direction onto the plane normal
    line_direction_on_plane_normal = np.dot(line_direction, plane_normal)

    # find scale factor for line direction
    scale_factor = line_plane_on_plane_normal / line_direction_on_plane_normal

    return line_position + (scale_factor * line_direction)
