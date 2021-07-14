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
