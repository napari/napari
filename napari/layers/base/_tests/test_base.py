from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pytest

from napari.layers.base._test_util_sample_layer import SampleLayer
from napari.layers.base.base import OPTIONAL_PARAMETERS
from napari.layers.utils.layer_utils import coerce_affine


def test_post_init():
    layer = SampleLayer(np.empty((10, 10)))
    assert layer.a == 1


def test_unset_params():
    layer = SampleLayer(np.empty((10, 10)))
    assert layer.parameters_with_default_values == OPTIONAL_PARAMETERS


@pytest.mark.parametrize(
    ('parameter', 'value'),
    [
        ('affine', coerce_affine(np.array([[0, -1], [1, 0]]), ndim=2)),
        ('rotate', [[0, -1], [1, 0]]),
        ('shear', [0.5]),
        ('scale', (1, 2)),
        ('translate', (1, 1)),
    ],
)
def test_set_parameter_simple(parameter, value):
    mock = Mock()
    layer = SampleLayer(np.empty((10, 10)), 2)
    getattr(layer.events, parameter).connect(mock)
    assert parameter in layer.parameters_with_default_values
    setattr(layer, parameter, value)
    assert parameter not in layer.parameters_with_default_values
    mock.assert_called_once()
    npt.assert_array_equal(getattr(layer, parameter), value)


def test_unset_scale():
    layer = SampleLayer(np.empty((10, 10)), scale=(1, 2))
    assert 'scale' not in layer.parameters_with_default_values
    layer.scale = None
    assert 'scale' in layer.parameters_with_default_values
    npt.assert_array_equal(layer.scale, (1, 1))
