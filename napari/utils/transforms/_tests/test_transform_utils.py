import numpy as np
import pytest

from napari.utils.transforms.transform_utils import (
    compose_linear_matrix,
    decompose_linear_matrix,
    is_matrix_lower_triangular,
    is_matrix_triangular,
    is_matrix_upper_triangular,
    shear_matrix_from_angle,
)


@pytest.mark.parametrize('upper_triangular', [True, False])
def test_decompose_linear_matrix(upper_triangular):
    """Test composing and decomposing a linear matrix."""
    np.random.seed(0)

    # Decompose linear matrix
    A = np.random.random((2, 2))
    rotate, scale, shear = decompose_linear_matrix(
        A, upper_triangular=upper_triangular
    )

    # Compose linear matrix and check it matches
    B = compose_linear_matrix(rotate, scale, shear)
    np.testing.assert_almost_equal(A, B)

    # Decompose linear matrix and check it matches
    rotate_B, scale_B, shear_B = decompose_linear_matrix(
        B, upper_triangular=upper_triangular
    )
    np.testing.assert_almost_equal(rotate, rotate_B)
    np.testing.assert_almost_equal(scale, scale_B)
    np.testing.assert_almost_equal(shear, shear_B)

    # Compose linear matrix and check it matches
    C = compose_linear_matrix(rotate_B, scale_B, shear_B)
    np.testing.assert_almost_equal(B, C)


@pytest.mark.parametrize('angle_degrees', range(-180, 180, 30))
def test_decompose_linear_matrix_with_pure_rotation(angle_degrees):
    # See the GitHub issue for more details:
    # https://github.com/napari/napari/issues/2984
    input_matrix = _make_2d_rotate_matrix(angle_degrees)
    rotate_output, scale, shear = decompose_linear_matrix(input_matrix)
    np.testing.assert_almost_equal(input_matrix, rotate_output)


def test_decompose_linear_matrix_with_rotation_and_reflection():
    # See the GitHub issue for more details:
    # https://github.com/napari/napari/issues/2984
    scale = [-1, 1]
    rotate = _make_2d_rotate_matrix(30)
    input_matrix = rotate * scale

    rotate_output, scale, shear = decompose_linear_matrix(input_matrix)

    np.testing.assert_almost_equal(input_matrix, rotate_output)


def _make_2d_rotate_matrix(angle_degrees):
    angle_radians = np.deg2rad(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    return np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])


def test_composition_order():
    """Test composition order."""
    # Order is rotate, shear, scale
    rotate = np.array([[0, -1], [1, 0]])
    shear = np.array([[1, 3], [0, 1]])
    scale = [2, 5]

    matrix = compose_linear_matrix(rotate, scale, shear)
    np.testing.assert_almost_equal(matrix, rotate @ shear @ np.diag(scale))


def test_shear_matrix_from_angle():
    """Test creating a shear matrix from an angle."""
    matrix = shear_matrix_from_angle(35)
    np.testing.assert_almost_equal(np.diag(matrix), [1] * 3)
    np.testing.assert_almost_equal(matrix[-1, 0], np.tan(np.deg2rad(55)))


upper = np.array([[1, 1], [0, 1]])
lower = np.array([[1, 0], [1, 1]])
full = np.array([[1, 1], [1, 1]])


def test_is_matrix_upper_triangular():
    """Test if a matrix is upper triangular."""
    assert is_matrix_upper_triangular(upper)
    assert not is_matrix_upper_triangular(lower)
    assert not is_matrix_upper_triangular(full)


def test_is_matrix_lower_triangular():
    """Test if a matrix is lower triangular."""
    assert not is_matrix_lower_triangular(upper)
    assert is_matrix_lower_triangular(lower)
    assert not is_matrix_lower_triangular(full)


def test_is_matrix_triangular():
    """Test if a matrix is triangular."""
    assert is_matrix_triangular(upper)
    assert is_matrix_triangular(lower)
    assert not is_matrix_triangular(full)
