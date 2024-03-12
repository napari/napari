from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pytest
import unyt

from napari.layers.utils.layer_utils import coerce_affine
from napari.layers.utils.spatial_information import (
    _OPTIONAL_PARAMETERS,
    SpatialInformation,
    _coerce_units_and_axes,
    _get_units_from_name,
)


class TestGetUnitsFromName:
    def test_simple(self):
        assert _get_units_from_name('m') == unyt.m
        assert _get_units_from_name('meter') == unyt.m
        assert _get_units_from_name(unyt.m) == unyt.m

    def test_none(self):
        assert _get_units_from_name(None) is None

    def test_dict(self):
        assert _get_units_from_name({'x': 'm', 'y': 'cm'}) == {
            'x': unyt.m,
            'y': unyt.cm,
        }
        assert _get_units_from_name({'x': unyt.m, 'y': 'cm'}) == {
            'x': unyt.m,
            'y': unyt.cm,
        }

    def test_exception(self):
        with pytest.raises(ValueError, match='Could not find unit bla'):
            _get_units_from_name('bla')


class TestCoerceUnitsAndAxes:
    def test_simple(self):
        assert _coerce_units_and_axes('m', None) == (unyt.m, None)
        assert _coerce_units_and_axes('m', ['x', 'y']) == (unyt.m, ['x', 'y'])

    def test_filter_units(self):
        assert _coerce_units_and_axes(
            {'x': 'm', 'y': 'cm', 'z': 'mm'}, ['x', 'y']
        ) == ({'x': unyt.m, 'y': unyt.cm}, ['x', 'y'])

    def test_no_uint_exception(self):
        with pytest.raises(
            ValueError,
            match='If both axes_labels and units are provided.* Missing units for: y',
        ):
            _coerce_units_and_axes({'x': 'm', 'z': 'cm'}, ['x', 'y'])

    def test_unique_axes_exception(self):
        with pytest.raises(ValueError, match='Axes labels must be unique'):
            _coerce_units_and_axes(
                {'x': 'm', 'y': 'cm', 'z': 'mm'}, ['x', 'x']
            )


class TestSpatialInformation:
    def test_simple(self):
        si = SpatialInformation(ndim=2)
        assert si.ndim == 2
        npt.assert_array_equal(si.scale, (1, 1))
        assert si.unset_parameters == set(_OPTIONAL_PARAMETERS)

    @pytest.mark.parametrize(
        'parameter,value',
        [
            ('affine', coerce_affine(np.array([[0, -1], [1, 0]]), ndim=2)),
            ('rotate', [[0, -1], [1, 0]]),
            ('shear', [0.5]),
            ('scale', (1, 2)),
            ('translate', (1, 1)),
            ('units', {'x': unyt.m, 'y': unyt.mm}),
            ('axes_labels', ['x', 'y']),
        ],
    )
    def test_set_parameter_simple(self, parameter, value):
        mock = Mock()
        si = SpatialInformation(ndim=2)
        getattr(si.events, parameter).connect(mock)
        assert parameter in si.unset_parameters
        setattr(si, parameter, value)
        assert parameter not in si.unset_parameters
        mock.assert_called_once()
        npt.assert_array_equal(mock.call_args[0], (value,))
        npt.assert_array_equal(getattr(si, parameter), value)

    def test_units(self):
        si = SpatialInformation(ndim=2, units='m', axes_labels=['x', 'y'])
        assert si.units == {'x': unyt.m, 'y': unyt.m}
        assert si.axes_labels == ['x', 'y']

        si.units = 'cm'
        assert si.units == {'x': unyt.cm, 'y': unyt.cm}

        si.units = {'x': unyt.m, 'y': unyt.cm, 'z': unyt.mm}
        assert si.units == {'x': unyt.m, 'y': unyt.cm}

    def test_units_exceptions(self):
        si = SpatialInformation(ndim=2, units='m', axes_labels=['x', 'y'])
        with pytest.raises(
            ValueError,
            match='If both axes_labels and units are provided.* Missing units for: y',
        ):
            si.units = {'x': unyt.m}

        with pytest.raises(
            ValueError,
            match='If both axes_labels and units are provided.* Missing units for: y',
        ):
            si.units = {'x': unyt.m, 'z': unyt.m}

    def test_axis_labels(self):
        si = SpatialInformation(ndim=2, units='m', axes_labels=['x', 'y'])
        assert si.units == {'x': unyt.m, 'y': unyt.m}
        assert si.axes_labels == ['x', 'y']

        si.axes_labels = ['a', 'b']
        assert si.axes_labels == ['a', 'b']
        assert si.units == {'a': unyt.m, 'b': unyt.m}

    def test_axis_labels_exceptions(self):
        si = SpatialInformation(ndim=2, units='m', axes_labels=['x', 'y'])
        with pytest.raises(ValueError, match='Axes labels must be unique'):
            si.axes_labels = ['x', 'x']
        with pytest.raises(
            ValueError,
            match=r'Length of axes_labels should be equal to ndim \(2\)',
        ):
            si.axes_labels = ['x']

    def test_axis_labels_exceptions_units_per_axis(self):
        si = SpatialInformation(
            ndim=2, units={'x': 'm', 'y': 'cm'}, axes_labels=['x', 'y']
        )
        with pytest.raises(
            ValueError, match='Units are set per axis and some of new'
        ):
            si.axes_labels = ['x', 'a']

    def test_set_axis_and_units(self):
        si = SpatialInformation(ndim=2)
        mock1 = Mock()
        mock2 = Mock()
        si.events.axes_labels.connect(mock1)
        si.events.units.connect(mock2)
        si.set_axis_and_units(['a', 'b'], 'm')
        mock1.assert_called_once_with(['a', 'b'])
        mock2.assert_called_once_with({'a': unyt.m, 'b': unyt.m})
