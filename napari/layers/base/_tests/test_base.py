from unittest.mock import Mock

import numpy as np
import pint
import pytest

from napari.layers.base._test_util_sample_layer import SampleLayer

REG = pint.get_application_registry()


def test_assign_units():
    layer = SampleLayer(np.empty((10, 10)))
    mock = Mock()
    layer.events.units.connect(mock)
    assert layer.units == (REG.pixel, REG.pixel)

    layer.units = ('nm', 'nm')
    mock.assert_called_once()
    mock.reset_mock()

    assert layer.units == (REG.nm, REG.nm)

    layer.units = (REG.mm, REG.mm)
    mock.assert_called_once()
    mock.reset_mock()

    assert layer.units == (REG.mm, REG.mm)

    layer.units = ('mm', 'mm')
    mock.assert_not_called()

    layer.units = 'km'
    mock.assert_called_once()
    mock.reset_mock()
    assert layer.units == (REG.km, REG.km)

    layer.units = None
    mock.assert_called_once()
    assert layer.units == (REG.pixel, REG.pixel)


def test_units_constructor():
    layer = SampleLayer(np.empty((10, 10)), units=('nm', 'nm'))
    assert layer.units == (REG.nm, REG.nm)

    layer = SampleLayer(np.empty((10, 10)), units=(REG.mm, REG.mm))
    assert layer.units == (REG.mm, REG.mm)

    layer = SampleLayer(np.empty((10, 10)), units=('mm', 'mm'))
    assert layer.units == (REG.mm, REG.mm)

    layer = SampleLayer(np.empty((10, 10)), units=None)
    assert layer.units == (REG.pixel, REG.pixel)


def test_assign_units_error():
    layer = SampleLayer(np.empty((10, 10)))
    with pytest.raises(ValueError, match='must have length ndim'):
        layer.units = ('m', 'm', 'm')

    with pytest.raises(ValueError, match='Could not find unit'):
        layer.units = ('ugh', 'ugh')

    with pytest.raises(ValueError, match='Could not find unit'):
        SampleLayer(np.empty((10, 10)), units=('ugh', 'ugh'))

    with pytest.raises(ValueError, match='must have length ndim'):
        SampleLayer(np.empty((10, 10)), units=('m', 'm', 'm'))


def test_axis_labels_assign():
    layer = SampleLayer(np.empty((10, 10)))
    mock = Mock()
    layer.events.axis_labels.connect(mock)
    assert layer.axis_labels == ('axis -2', 'axis -1')

    layer.axis_labels = ('x', 'y')
    mock.assert_called_once()
    mock.reset_mock()

    assert layer.axis_labels == ('x', 'y')

    layer.axis_labels = ('x', 'y')
    mock.assert_not_called()

    layer.axis_labels = None
    mock.assert_called_once()
    assert layer.axis_labels == ('axis -2', 'axis -1')


def test_axis_labels_constructor():
    layer = SampleLayer(np.empty((10, 10)), axis_labels=('x', 'y'))
    assert layer.axis_labels == ('x', 'y')

    layer = SampleLayer(np.empty((10, 10)), axis_labels=None)
    assert layer.axis_labels == ('axis -2', 'axis -1')


def test_axis_labels_error():
    layer = SampleLayer(np.empty((10, 10)))
    with pytest.raises(ValueError, match='must have length ndim'):
        layer.axis_labels = ('x', 'y', 'z')

    with pytest.raises(ValueError, match='must have length ndim'):
        SampleLayer(np.empty((10, 10)), axis_labels=('x', 'y', 'z'))


def test_non_visible_mode():
    layer = SampleLayer(np.empty((10, 10)))
    layer.mode = 'transform'

    # change layer visibility and check the layer mode gets updated
    layer.visible = False
    assert layer.mode == 'pan_zoom'
    layer.visible = True
    assert layer.mode == 'transform'


def test_world_to_displayed_data_normal_3D():
    layer = SampleLayer(np.empty((10, 10, 10)))
    layer.scale = (1, 3, 2)

    normal_vector = [0, 1, 1]

    expected_transformed_vector = [0, 3 * (13**0.5) / 13, 2 * (13**0.5) / 13]

    transformed_vector = layer._world_to_displayed_data_normal(
        normal_vector, dims_displayed=[0, 1, 2]
    )

    assert np.allclose(transformed_vector, expected_transformed_vector)


def test_world_to_displayed_data_normal_4D():
    layer = SampleLayer(np.empty((10, 10, 10, 10)))
    layer.scale = (1, 3, 2, 1)

    normal_vector = [0, 1, 1]

    expected_transformed_vector = [0, 3 * (13**0.5) / 13, 2 * (13**0.5) / 13]

    transformed_vector = layer._world_to_displayed_data_normal(
        normal_vector, dims_displayed=[0, 1, 2]
    )

    assert np.allclose(transformed_vector, expected_transformed_vector)
