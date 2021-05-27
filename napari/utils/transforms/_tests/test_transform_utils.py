import numpy as np

from napari.utils.transforms.transform_utils import (
    is_matrix_lower_triangular,
    is_matrix_triangular,
    is_matrix_upper_triangular,
    shear_matrix_from_angle,
)


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
