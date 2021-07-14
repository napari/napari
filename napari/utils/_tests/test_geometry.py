import numpy as np
import pytest

from ..geometry import project_point_to_plane, rotation_matrix_from_vectors

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
    projected_point = project_point_to_plane(point, plane_point, plane_normal)

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
    rotation_matrix = rotation_matrix_from_vectors(vec_1, vec_2)
    rotated_1 = rotation_matrix.dot(vec_1)

    unit_rotated_1 = rotated_1 / np.linalg.norm(rotated_1)
    unit_vec_2 = vec_2 / np.linalg.norm(vec_2)

    np.testing.assert_allclose(unit_rotated_1, unit_vec_2)
