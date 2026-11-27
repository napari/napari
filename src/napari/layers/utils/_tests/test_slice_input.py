import numpy as np
import pint
import pytest

from napari.layers.utils._slice_input import (
    _SliceInput,
    _ThickNDSlice,
    units_conversion_factors,
)
from napari.utils.transforms import Affine


def _slice_input(ndim=3, ndisplay=2):
    return _SliceInput(
        ndisplay=ndisplay,
        world_slice=_ThickNDSlice.make_full(ndim=ndim),
        order=tuple(range(ndim)),
    )


def _rotation_matrix(angle_deg, axis0, axis1):
    angle = angle_deg/180*np.pi
    cos, sin = np.cos(angle), np.sin(angle)
    matrix = np.zeros((3,3))
    matrix[axis0, axis0] = cos
    matrix[axis0, axis1] = -sin
    matrix[axis1, axis0] = sin
    matrix[axis1, axis1] = cos
    return matrix


@pytest.mark.parametrize(
    'linear_matrix, expected',
    [
        (np.eye(3), True),
        (np.diag([1.0, 2.0, 0.5]), True),
        # rotation within the displayed plane
        (_rotation_matrix(30, 1, 2), True),
        # rotation tilting the slice plane out of the displayed axes
        (_rotation_matrix(30, 0, 1), False),
        # shear from a displayed world axis into a non-displayed data axis:
        # moving within the world slice changes a non-displayed data
        # coordinate, so the data planes are tilted
        (np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), False),
        # shear from a non-displayed world axis into a displayed data axis:
        # the world planes stay axis-aligned in data space
        (np.array([[1.0, 0.0, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]]), True),
    ],
)
def test_is_orthogonal(linear_matrix, expected):
    world_to_data = Affine(ndim=3, linear_matrix=linear_matrix, translate=None)
    assert _slice_input().is_orthogonal(world_to_data) is expected


def test_units_conversion_factors_without_world_units():
    reg = pint.get_application_registry()
    layer_units = (reg.micrometer,) * 3
    np.testing.assert_array_equal(
        units_conversion_factors(layer_units, None), np.ones(3)
    )


def test_units_conversion_factors_different_units():
    """Numbers convert as base(layer) / base(world): a length of one
    micrometer is one thousandth of a millimeter."""
    reg = pint.get_application_registry()
    layer_units = (reg.micrometer,) * 3
    factors = units_conversion_factors(layer_units, (reg.millimeter,) * 3)
    np.testing.assert_allclose(factors, 1e-3)
