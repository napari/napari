from typing import Dict

import numpy as np

# normal vectors for a 3D axis-aligned box
# coordinates are ordered [z, y, x]
FACE_NORMALS = {
    "x_pos": np.array([0, 0, 1]),
    "x_neg": np.array([0, 0, -1]),
    "y_pos": np.array([0, 1, 0]),
    "y_neg": np.array([0, -1, 0]),
    "z_pos": np.array([1, 0, 0]),
    "z_neg": np.array([-1, 0, 0]),
}


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
    """Find the intersection of a line with an axis aligned plane.

    Parameters
    ----------
    plane_intercept: float
        The coordinate that the plane intersects on the axis to which plane is
        normal.
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
    return line_start + t * line_direction


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
    """Find the intersection of a line with an arbitrarily oriented plane in 3D.
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


def point_in_quadrilateral_2d(
    point: np.ndarray, quadrilateral: np.ndarray
) -> bool:
    """Determines whether a point is inside a 2D quadrilateral.

    Parameters
    ----------
    point : np.ndarray
        (2,) array containing coordinates of a point.
    quadrilateral : np.ndarray
        (4, 2) array containing the coordinates for the 4 corners
        of a quadrilateral. The vertices should be in clockwise order
        such that indexing with [0, 1, 2], and [0, 2, 3] results in
        the two non-overlapping triangles that divide the
        quadrilateral.

    Returns
    -------

    """
    triangle_vertices = np.stack(
        (quadrilateral[[0, 1, 2]], quadrilateral[[0, 2, 3]])
    )
    in_triangles = inside_triangles(triangle_vertices - point)
    if in_triangles.sum() < 1:
        return False
    else:
        return True


def click_in_quadrilateral_3d(
    click_pos_data: np.ndarray,
    quadrilateral: np.ndarray,
    view_dir_data: np.ndarray,
) -> bool:
    """Determine if a click occurred within a specified quadrilateral.
    For example, this could be used to determine if a click was
    in a specific face of a bounding box.

    Parameters
    ----------
    click_pos_data : np.ndarray
        (3,) array containing the location that was clicked. This
        should be in the same coordinate system as the vertices.
    quadrilateral : np.ndarray
        (4, 3) array containing the coordinates for the 4 corners
        of a quadrilateral. The vertices should be in clockwise order
        such that indexing with [0, 1, 2], and [0, 2, 3] results in
        the two triangles non-overlapping triangles that divide the
        quadrilateral.
    view_dir_data : np.ndarray
        (3,) array describing the direction camera is pointing in
        the scene. This should be in the same coordinate system as
        the vertices.

    Returns
    -------
    in_region : bool
        True if the click is in the region specified by vertices.
    """

    # project the vertices on to the view plane
    vertices_plane = project_point_to_plane(
        point=quadrilateral,
        plane_point=click_pos_data,
        plane_normal=view_dir_data,
    )

    # rotate the plane to make the triangles 2D
    rotation_matrix = rotation_matrix_from_vectors(view_dir_data, [0, 0, 1])
    rotated_vertices = vertices_plane @ rotation_matrix.T
    vertices_2D = rotated_vertices[:, :2]
    click_pos_2D = rotation_matrix.dot(click_pos_data)[:2]

    quadrilateral = np.stack((vertices_2D[[0, 1, 2]], vertices_2D[[0, 2, 3]]))
    return point_in_quadrilateral_2d(click_pos_2D, quadrilateral)


def find_front_back_face(
    click_pos: np.ndarray, bounding_box: np.ndarray, view_dir: np.ndarray
):
    """Find the faces of an axis aligned bounding box a
    click intersects with.

    Parameters
    ----------
    click_pos : np.ndarray
        (3,) array containing the location that was clicked.
    bounding_box : np.ndarray
        (N, 2), N=ndim array with the min and max value for each dimension of
        the bounding box. The bounding box is take form the last
        three rows, which are assumed to be in order (z, y, x).
        This should be in the same coordinate system as click_pos.
    view_dir
        (3,) array describing the direction camera is pointing in
        the scene. This should be in the same coordinate system as click_pos.

    Returns
    -------
    front_face_normal : np.ndarray
        The (3,) normal vector of the face closest to the camera the click
        intersects with.
    back_face_normal : np.ndarray
        The (3,) normal vector of the face farthest from the camera the click
        intersects with.
    """
    front_face_normal = None
    back_face_normal = None

    bbox_face_coords = bounding_box_to_face_vertices(bounding_box)
    for k, v in FACE_NORMALS.items():
        if (np.dot(view_dir, v) + 0.001) < 0:
            if click_in_quadrilateral_3d(
                click_pos, bbox_face_coords[k], view_dir
            ):
                front_face_normal = v
        elif (np.dot(view_dir, v) + 0.001) > 0:
            if click_in_quadrilateral_3d(
                click_pos, bbox_face_coords[k], view_dir
            ):
                back_face_normal = v
        if front_face_normal is not None and back_face_normal is not None:
            # stop looping if both the front and back faces have been found
            break

    return front_face_normal, back_face_normal


def find_click_bbox_face_intersection(
    face_normal,
    click_pos: np.ndarray,
    bounding_box: np.ndarray,
    view_dir: np.ndarray,
):
    """Find the locations of where a click intersects with the
    specified face of an axis-aligned bounding box

    Parameters
    ----------
    face_normal : np.ndarray
        The (3,) normal vector of the face the click intersects with.
    click_pos : np.ndarray
        (3,) array containing the location that was clicked.
    bounding_box : np.ndarray
        (N, 2), N=ndim array with the min and max value for each dimension of
        the bounding box. The bounding box is take form the last
        three rows, which are assumed to be in order (z, y, x).
        This should be in the same coordinate system as click_pos.
    view_dir
        (3,) array describing the direction camera is pointing in
        the scene. This should be in the same coordinate system as click_pos.

    Returns
    -------
    intersection_point : np.ndarray
        (3,) array containing the coordinate for the intersection of the click on
        the specified face.
    """
    front_face_coordinate = face_coordinate_from_bounding_box(
        bounding_box, face_normal
    )
    intersection_point = np.squeeze(
        intersect_line_with_axis_aligned_plane(
            front_face_coordinate,
            face_normal,
            click_pos,
            -view_dir,
        )
    )

    return intersection_point


def distance_between_point_and_line_3d(
    point: np.ndarray, line_position: np.ndarray, line_direction: np.ndarray
):
    """Determine the minimum distance between a point and a line in 3D.

    Parameters
    ----------
    point : np.ndarray
        (3,) array containing coordinates of a point in 3D space.
    line_position : np.ndarray
        (3,) array containing coordinates of a point on a line in 3D space.
    line_direction : np.ndarray
        (3,) array containing a vector describing the direction of a line in
        3D space.

    Returns
    -------
    distance : float
        The minimum distance between `point` and the line defined by
        `line_position` and `line_direction`.
    """
    line_direction_normalized = line_direction / np.linalg.norm(line_direction)
    projection_on_line_direction = np.dot(
        (point - line_position), line_direction
    )
    closest_point_on_line = (
        line_position
        + line_direction_normalized * projection_on_line_direction
    )
    distance = np.linalg.norm(point - closest_point_on_line)
    return distance
