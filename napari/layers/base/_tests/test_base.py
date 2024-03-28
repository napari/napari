from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pint
import pytest

from napari.layers.base.base import Layer
from napari.layers.utils.layer_utils import coerce_affine


class SampleLayer(Layer):

    def __init__(
        self,
        data,
        ndim,
        *,
        affine=None,
        axes_labels=None,
        blending='translucent',
        cache=True,  # this should move to future "data source" object.
        experimental_clipping_planes=None,
        metadata=None,
        mode='pan_zoom',
        multiscale=False,
        name=None,
        opacity=1.0,
        projection_mode='none',
        rotate=None,
        scale=None,
        shear=None,
        translate=None,
        units=None,
        visible=True,
    ):
        super().__init__(
            ndim=ndim,
            data=data,
            affine=affine,
            axes_labels=axes_labels,
            blending=blending,
            cache=cache,
            experimental_clipping_planes=experimental_clipping_planes,
            metadata=metadata,
            mode=mode,
            multiscale=multiscale,
            name=name,
            opacity=opacity,
            projection_mode=projection_mode,
            rotate=rotate,
            scale=scale,
            shear=shear,
            translate=translate,
            units=units,
            visible=visible,
        )
        self._data = data
        self.a = 2

    @property
    def data(self):
        return self._data

    @property
    def _extent_data(self) -> np.ndarray:
        pass

    def _get_ndim(self) -> int:
        return self.ndim

    def _get_state(self):
        base_state = self._get_base_state()
        base_state['data'] = self.data
        return base_state

    def _set_view_slice(self):
        pass

    def _update_thumbnail(self):
        pass

    def _get_value(self, position):
        return self.data[position]

    def _post_init(self):
        self.a = 1


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
    si = SampleLayer(np.zeros((10, 10)), 2, units='m', axes_labels=['x', 'y'])
    assert si.units == {'x': unit_register.m, 'y': unit_register.m}
    assert si.axes_labels == ['x', 'y']

    si.units = 'cm'
    assert si.units == {'x': unit_register.cm, 'y': unit_register.cm}

    si.units = {
        'x': unit_register.m,
        'y': unit_register.cm,
        'z': unit_register.mm,
    }
    assert si.units == {'x': unit_register.m, 'y': unit_register.cm}


def test_units_exceptions(unit_register):
    si = SampleLayer(np.zeros((10, 10)), 2, units='m', axes_labels=['x', 'y'])
    with pytest.raises(
        ValueError,
        match='If both axes_labels and units are provided.* Missing units for: y',
    ):
        si.units = {'x': unit_register.m}

    with pytest.raises(
        ValueError,
        match='If both axes_labels and units are provided.* Missing units for: y',
    ):
        si.units = {'x': unit_register.m, 'z': unit_register.m}


def test_axis_labels(unit_register):
    si = SampleLayer(np.zeros((10, 10)), 2, units='m', axes_labels=['x', 'y'])
    assert si.units == {'x': unit_register.m, 'y': unit_register.m}
    assert si.axes_labels == ['x', 'y']

    si.axes_labels = ['a', 'b']
    assert si.axes_labels == ['a', 'b']
    assert si.units == {'a': unit_register.m, 'b': unit_register.m}


def test_axis_labels_exceptions():
    si = SampleLayer(np.zeros((10, 10)), 2, units='m', axes_labels=['x', 'y'])
    with pytest.raises(ValueError, match='Axes labels must be unique'):
        si.axes_labels = ['x', 'x']
    with pytest.raises(
        ValueError,
        match=r'Length of axes_labels should be equal to ndim \(2\)',
    ):
        si.axes_labels = ['x']


def test_axis_labels_exceptions_units_per_axis():
    si = SampleLayer(
        np.zeros((10, 10)),
        2,
        units={'x': 'm', 'y': 'cm'},
        axes_labels=['x', 'y'],
    )
    with pytest.raises(
        ValueError, match='Units are set per axis and some of new'
    ):
        si.axes_labels = ['x', 'a']


def test_set_axis_and_units(unit_register):
    si = SampleLayer(np.zeros((10, 10)), 2)
    mock1 = Mock()
    mock2 = Mock()
    si.events.axes_labels.connect(mock1)
    si.events.units.connect(mock2)
    si.set_axis_and_units(['a', 'b'], 'm')
    mock1.assert_called_once()
    mock2.assert_called_once()
    assert si.axes_labels == ['a', 'b']
    assert si.units == {'a': unit_register.m, 'b': unit_register.m}
