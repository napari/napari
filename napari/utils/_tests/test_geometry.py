import numpy as np
import pytest

from ..geometry import (
    bounding_box_to_face_vertices,
    clamp_point_to_bounding_box,
    click_in_quadrilateral_3d,
    face_coordinate_from_bounding_box,
    inside_triangles,
    intersect_line_with_axis_aligned_plane,
    intersect_line_with_plane_3d,
    point_in_quadrilateral_2d,
    project_point_onto_plane,
    rotation_matrix_from_vectors,
)

single_point = np.array([10, 10, 10])
expected_point_single = np.array([[10, 0, 10]])
multiple_point = np.array([[10, 10, 10], [20, 10, 30], [20, 40, 20]])
expected_multiple_point = np.array([[10, 0, 10], [20, 0, 30], [20, 0, 20]])


@pytest.mark.parametrize(
    "point,expected_projected_point",
    [
        (single_point, expected_point_single),
        (multiple_point, expected_multiple_point),
    ],
)
def test_project_point_to_plane(point, expected_projected_point):
    plane_point = np.array([20, 0, 0])
    plane_normal = np.array([0, 1, 0])
    projected_point = project_point_onto_plane(
        point, plane_point, plane_normal
    )

    np.testing.assert_allclose(projected_point, expected_projected_point)


@pytest.mark.parametrize(
    "vec_1, vec_2",
    [
        (np.array([10, 0, 0]), np.array([0, 5, 0])),
        (np.array([0, 5, 0]), np.array([0, 5, 0])),
        (np.array([0, 5, 0]), np.array([0, -5, 0])),
    ],
)
def test_rotation_matrix_from_vectors(vec_1, vec_2):
    """Test that calculated rotation matrices align vec1 to vec2."""
    rotation_matrix = rotation_matrix_from_vectors(vec_1, vec_2)

    rotated_1 = rotation_matrix.dot(vec_1)
    unit_rotated_1 = rotated_1 / np.linalg.norm(rotated_1)

    unit_vec_2 = vec_2 / np.linalg.norm(vec_2)

    np.testing.assert_allclose(unit_rotated_1, unit_vec_2)


@pytest.mark.parametrize(
    "line_position, line_direction, plane_position, plane_normal, expected",
    [
        ([0, 0, 1], [0, 0, -1], [0, 0, 0], [0, 0, 1], [0, 0, 0]),
        ([1, 1, 1], [-1, -1, -1], [0, 0, 0], [0, 0, 1], [0, 0, 0]),
        ([2, 2, 2], [-1, -1, -1], [1, 1, 1], [0, 0, 1], [1, 1, 1]),
    ],
)
def test_intersect_line_with_plane_3d(
    line_position, line_direction, plane_position, plane_normal, expected
):
    """Test that arbitrary line-plane intersections are correctly calculated."""
    intersection = intersect_line_with_plane_3d(
        line_position, line_direction, plane_position, plane_normal
    )
    np.testing.assert_allclose(expected, intersection)


@pytest.mark.parametrize(
    "point, bounding_box, expected",
    [
        ([5, 5, 5], np.array([[0, 10], [0, 10], [0, 10]]), [5, 5, 5]),
        ([10, 10, 10], np.array([[0, 10], [0, 10], [0, 10]]), [9, 9, 9]),
        ([5, 5, 15], np.array([[0, 10], [0, 10], [0, 10]]), [5, 5, 9]),
    ],
)
def test_clamp_point_to_bounding_box(point, bounding_box, expected):
    """Test that points are correctly clamped to the limits of the data.
    Note: bounding boxes are calculated from layer extents, points are clamped
    to the range of valid indices into each dimension.

    e.g. for a shape (10,) array, data is clamped to the range (0, 9)
    """
    clamped_point = clamp_point_to_bounding_box(point, bounding_box)
    np.testing.assert_allclose(expected, clamped_point)


@pytest.mark.parametrize(
    'bounding_box, face_normal, expected',
    [
        (np.array([[5, 10], [10, 20], [20, 30]]), np.array([1, 0, 0]), 10),
        (np.array([[5, 10], [10, 20], [20, 30]]), np.array([-1, 0, 0]), 5),
        (np.array([[5, 10], [10, 20], [20, 30]]), np.array([0, 1, 0]), 20),
        (np.array([[5, 10], [10, 20], [20, 30]]), np.array([0, -1, 0]), 10),
        (np.array([[5, 10], [10, 20], [20, 30]]), np.array([0, 0, 1]), 30),
        (np.array([[5, 10], [10, 20], [20, 30]]), np.array([0, 0, -1]), 20),
    ],
)
def test_face_coordinate_from_bounding_box(
    bounding_box, face_normal, expected
):
    """Test that the correct face coordinate is calculated.

    Face coordinate is a float which is the value where a face of a bounding box,
    defined by a face normal, intersects the axis the normal vector is aligned with.
    """
    face_coordinate = face_coordinate_from_bounding_box(
        bounding_box, face_normal
    )
    np.testing.assert_allclose(expected, face_coordinate)


@pytest.mark.parametrize(
    'plane_intercept, plane_normal, line_start, line_direction, expected',
    [
        (
            0,
            np.array([0, 0, 1]),
            np.array([0, 0, 1]),
            np.array([0, 0, 1]),
            [0, 0, 0],
        ),
        (
            10,
            np.array([0, 0, 1]),
            np.array([0, 0, 0]),
            np.array([0, 0, 1]),
            [0, 0, 10],
        ),
        (
            10,
            np.array([0, 1, 0]),
            np.array([0, 1, 0]),
            np.array([0, 1, 0]),
            [0, 10, 0],
        ),
        (
            10,
            np.array([1, 0, 0]),
            np.array([1, 0, 0]),
            np.array([1, 0, 0]),
            [10, 0, 0],
        ),
    ],
)
def test_line_with_axis_aligned_plane(
    plane_intercept, plane_normal, line_start, line_direction, expected
):
    """Test that intersections between line and axis aligned plane are
    calculated correctly.
    """
    intersection = intersect_line_with_axis_aligned_plane(
        plane_intercept, plane_normal, line_start, line_direction
    )
    np.testing.assert_allclose(expected, intersection)


def test_bounding_box_to_face_vertices_3d():
    """Test that bounding_box_to_face_vertices returns a dictionary of vertices
    for each face of an axis aligned 3D bounding box.
    """
    bounding_box = np.array([[5, 10], [15, 20], [25, 30]])
    face_vertices = bounding_box_to_face_vertices(bounding_box)
    expected = {
        'x_pos': np.array(
            [[5, 15, 30], [5, 20, 30], [10, 20, 30], [10, 15, 30]]
        ),
        'x_neg': np.array(
            [[5, 15, 25], [5, 20, 25], [10, 20, 25], [10, 15, 25]]
        ),
        'y_pos': np.array(
            [[5, 20, 25], [5, 20, 30], [10, 20, 30], [10, 20, 25]]
        ),
        'y_neg': np.array(
            [[5, 15, 25], [5, 15, 30], [10, 15, 30], [10, 15, 25]]
        ),
        'z_pos': np.array(
            [[10, 15, 25], [10, 15, 30], [10, 20, 30], [10, 20, 25]]
        ),
        'z_neg': np.array(
            [[5, 15, 25], [5, 15, 30], [5, 20, 30], [5, 20, 25]]
        ),
    }
    for k in face_vertices:
        np.testing.assert_allclose(expected[k], face_vertices[k])


def test_bounding_box_to_face_vertices_nd():
    """Test that bounding_box_to_face_vertices returns a dictionary of vertices
    for each face of an axis aligned nD bounding box.
    """
    bounding_box = np.array([[0, 0], [0, 0], [5, 10], [15, 20], [25, 30]])
    face_vertices = bounding_box_to_face_vertices(bounding_box)
    expected = {
        'x_pos': np.array(
            [[5, 15, 30], [5, 20, 30], [10, 20, 30], [10, 15, 30]]
        ),
        'x_neg': np.array(
            [[5, 15, 25], [5, 20, 25], [10, 20, 25], [10, 15, 25]]
        ),
        'y_pos': np.array(
            [[5, 20, 25], [5, 20, 30], [10, 20, 30], [10, 20, 25]]
        ),
        'y_neg': np.array(
            [[5, 15, 25], [5, 15, 30], [10, 15, 30], [10, 15, 25]]
        ),
        'z_pos': np.array(
            [[10, 15, 25], [10, 15, 30], [10, 20, 30], [10, 20, 25]]
        ),
        'z_neg': np.array(
            [[5, 15, 25], [5, 15, 30], [5, 20, 30], [5, 20, 25]]
        ),
    }
    for k in face_vertices:
        np.testing.assert_allclose(expected[k], face_vertices[k])


@pytest.mark.parametrize(
    'triangle, expected',
    [
        (np.array([[[-1, -1], [-1, 1], [1, 0]]]), True),
        (np.array([[[1, 1], [2, 1], [1.5, 2]]]), False),
    ],
)
def test_inside_triangles(triangle, expected):
    """Test that inside triangles returns an array of True for triangles which
    contain the origin, False otherwise.
    """
    inside = np.all(inside_triangles(triangle))
    assert inside == expected


@pytest.mark.parametrize(
    'point, quadrilateral, expected',
    [
        (
            np.array([0.5, 0.5]),
            np.array([[0, 0], [0, 1], [1, 1], [0, 1]]),
            True,
        ),
        (np.array([2, 2]), np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), False),
    ],
)
def test_point_in_quadrilateral_2d(point, quadrilateral, expected):
    """Test that point_in_quadrilateral_2d determines whether a point
    is inside a quadrilateral.
    """
    inside = point_in_quadrilateral_2d(point, quadrilateral)
    assert inside == expected


@pytest.mark.parametrize(
    'click_position, quadrilateral, view_dir, expected',
    [
        (
            np.array([0, 0, 0]),
            np.array([[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]]),
            np.array([0, 0, 1]),
            True,
        ),
        (
            np.array([0, 0, 5]),
            np.array([[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]]),
            np.array([0, 0, 1]),
            True,
        ),
        (
            np.array([0, 5, 0]),
            np.array([[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]]),
            np.array([0, 0, 1]),
            False,
        ),
    ],
)
def test_click_in_quadrilateral_3d(
    click_position, quadrilateral, view_dir, expected
):
    """Test that click in quadrilateral 3d determines whether the projection
    of a 3D point onto a plane falls within a 3d quadrilateral projected
    onto the same plane
    """
    in_quadrilateral = click_in_quadrilateral_3d(
        click_position, quadrilateral, view_dir
    )
    assert in_quadrilateral == expected
