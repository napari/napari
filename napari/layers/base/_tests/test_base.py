from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pint
import pytest

from napari.layers.base._test_util_samlpe_layer import SampleLayer
from napari.layers.utils.layer_utils import coerce_affine


def test_post_init():
    layer = SampleLayer(np.zeros((5, 10, 10)), 3)
    assert layer.a == 1


def test_sample_layer(unit_register):
    layer = SampleLayer(np.zeros((5, 10, 10)), 3)
    assert layer.ndim == 3
    assert layer.axes_labels == [f'dim_{i}' for i in range(3)][::-1]
    assert layer.units == {f'dim_{i}': unit_register.pixel for i in range(3)}
    npt.assert_array_equal(layer.scale, [1, 1, 1])
    npt.assert_array_equal(layer.translate, [0, 0, 0])


def test_set_units(unit_register):
    layer = SampleLayer(np.zeros((5, 10, 10)), 3, units='nm')
    assert layer.units == {
        f'dim_{i}': unit_register.nanometer for i in range(3)
    }
    assert 'units' not in layer.parameters_with_default_values


@pytest.mark.parametrize(
    'parameter,value',
    [
        ('affine', coerce_affine(np.array([[0, -1], [1, 0]]), ndim=2)),
        ('rotate', [[0, -1], [1, 0]]),
        ('shear', [0.5]),
        ('scale', (1, 2)),
        ('translate', (1, 1)),
        ('units', {'dim_1': pint.Unit('m'), 'dim_0': pint.Unit('mm')}),
        ('axes_labels', ['x', 'y']),
    ],
)
def test_set_parameter_simple(parameter, value):
    mock = Mock()
    layer = SampleLayer(np.zeros((10, 10)), 2)
    getattr(layer.events, parameter).connect(mock)
    assert parameter in layer.parameters_with_default_values
    setattr(layer, parameter, value)
    assert parameter not in layer.parameters_with_default_values
    mock.assert_called_once()
    npt.assert_array_equal(getattr(layer, parameter), value)


def test_units(unit_register):
    layer = SampleLayer(
        np.zeros((10, 10)), 2, units='m', axes_labels=['x', 'y']
    )
    assert layer.units == {'x': unit_register.m, 'y': unit_register.m}
    assert layer.axes_labels == ['x', 'y']

    layer.units = 'cm'
    assert layer.units == {'x': unit_register.cm, 'y': unit_register.cm}

    layer.units = {
        'x': unit_register.m,
        'y': unit_register.cm,
        'z': unit_register.mm,
    }
    assert layer.units == {'x': unit_register.m, 'y': unit_register.cm}


def test_units_exceptions(unit_register):
    layer = SampleLayer(
        np.zeros((10, 10)), 2, units='m', axes_labels=['x', 'y']
    )
    with pytest.raises(
        ValueError,
        match='If both axes_labels and units are provided.* Missing units for: y',
    ):
        layer.units = {'x': unit_register.m}

    with pytest.raises(
        ValueError,
        match='If both axes_labels and units are provided.* Missing units for: y',
    ):
        layer.units = {'x': unit_register.m, 'z': unit_register.m}


def test_axis_labels(unit_register):
    layer = SampleLayer(
        np.zeros((10, 10)), 2, units='m', axes_labels=['x', 'y']
    )
    assert layer.units == {'x': unit_register.m, 'y': unit_register.m}
    assert layer.axes_labels == ['x', 'y']

    layer.axes_labels = ['a', 'b']
    assert layer.axes_labels == ['a', 'b']
    assert layer.units == {'a': unit_register.m, 'b': unit_register.m}


def test_axis_labels_exceptions():
    layer = SampleLayer(
        np.zeros((10, 10)), 2, units='m', axes_labels=['x', 'y']
    )
    with pytest.raises(ValueError, match='Axes labels must be unique'):
        layer.axes_labels = ['x', 'x']
    with pytest.raises(
        ValueError,
        match=r'Length of axes_labels should be equal to ndim \(2\)',
    ):
        layer.axes_labels = ['x']


def test_axis_labels_exceptions_units_per_axis():
    layer = SampleLayer(
        np.zeros((10, 10)),
        2,
        units={'x': 'm', 'y': 'cm'},
        axes_labels=['x', 'y'],
    )
    with pytest.raises(
        ValueError, match='Units are set per axis and some of new'
    ):
        layer.axes_labels = ['x', 'a']


def test_set_axis_and_units(unit_register):
    layer = SampleLayer(np.zeros((10, 10)), 2)
    mock1 = Mock()
    mock2 = Mock()
    layer.events.axes_labels.connect(mock1)
    layer.events.units.connect(mock2)
    layer.set_axis_and_units(['a', 'b'], 'm')
    mock1.assert_called_once()
    mock2.assert_called_once()
    assert layer.axes_labels == ['a', 'b']
    assert layer.units == {'a': unit_register.m, 'b': unit_register.m}


def test_axis_from_units(unit_register):
    layer = SampleLayer(np.zeros((10, 10)), units={'a': 'm', 'b': 's'})
    assert layer.axes_labels == ['a', 'b']
    assert layer.units == {'a': unit_register.m, 'b': unit_register.s}


def test_axis_from_units_setter(unit_register):
    layer = SampleLayer(np.zeros((10, 10)))
    layer.units = {'a': 'm', 'b': 's'}
    assert layer.axes_labels == ['a', 'b']
    assert layer.units == {'a': unit_register.m, 'b': unit_register.s}
    with pytest.raises(ValueError, match='If both'):
        layer.units = {'x': 'm', 'y': 's'}
