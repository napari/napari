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
    """

    Parameters
    ----------
    point : np.ndarray
        n-dimensional point as an (n,) ndarray
    bounding_box: np.ndarray
        n-dimensional bounding box as a (2, n) ndarray

    Returns
    -------
    clamped_point : np.ndarray
        `point` clamped to the limits of `bounding_box`

    """
    clamped_point = np.clip(point, bounding_box[:, 0], bounding_box[:, 1] - 1)
    return clamped_point


def face_intercepts_from_bounding_box(bbox: np.ndarray):
    """TODO: kevin, can you remember what this does?"""
    face_intercepts = {k: bbox[v[0], v[1]] for k, v in FACE_INTERCEPTS.items()}
    return face_intercepts


def intersect_line_with_axis_aligned_plane(
    face_intercept: float,
    face_normal: np.ndarray,
    line_start: np.ndarray,
    line_direction: np.ndarray,
) -> np.ndarray:
    """
    Find the intersection of a ray with an axis aligned plane

    Parameters
    ----------
    face_intercept: TODO: kevin?
    face_normal : np.ndarray
        normal vector of the face as an (n,)  ndarray
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
    plane_axis = np.squeeze(np.argwhere(face_normal))

    # get the intersection coordinate
    t = (face_intercept - line_start[plane_axis]) / line_direction[plane_axis]
    intersection_point = line_start + t * line_direction

    return intersection_point


def bounding_box_to_face_vertices(bounding_box: np.ndarray) -> dict:
    """
    From a layer bounding box (N, 2), N=ndim, return a dictionary containing
    the vertices of each face of the bounding_box
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


FACE_INTERCEPTS = {
    "x_pos": [2, 1],
    "x_neg": [2, 0],
    "y_pos": [1, 1],
    "y_neg": [1, 0],
    "z_pos": [0, 1],
    "z_neg": [0, 0],
}
