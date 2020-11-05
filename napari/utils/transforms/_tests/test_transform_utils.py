import numpy as np
import pytest

from napari.utils.transforms.transform_utils import (
    check_shear_triangular,
    compose_linear_matrix,
    decompose_linear_matrix,
    is_lower_triangular,
    is_upper_triangular,
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


def test_shear_matrix_from_angle():
    """Test creating a shear matrix from an angle."""
    matrix = shear_matrix_from_angle(35)
    np.testing.assert_almost_equal(np.diag(matrix), [1] * 3)
    np.testing.assert_almost_equal(matrix[-1, 0], np.tan(np.deg2rad(55)))


upper = np.array([[1, 1], [0, 1]])
lower = np.array([[1, 0], [1, 1]])
full = np.array([[1, 1], [1, 1]])


def test_is_upper_triangular():
    """Test if a matrix is upper triangular."""
    assert is_upper_triangular(upper)
    assert not is_upper_triangular(lower)
    assert not is_upper_triangular(full)


def test_is_lower_triangular():
    """Test if a matrix is lower triangular."""
    assert not is_lower_triangular(upper)
    assert is_lower_triangular(lower)
    assert not is_lower_triangular(full)


def test_check_shear_triangular():
    """Determine shear triangular of matrix."""
    assert check_shear_triangular(upper)
    assert not check_shear_triangular(lower)
    with pytest.raises(ValueError):
        check_shear_triangular(full)
