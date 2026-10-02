import numpy as np
import pytest
import xarray as xr

from napari.utils._xarray_utils import (
    _check_xarray,
    _coord_metadata,
    _CoordMetadata,
    _get_xr_metadata,
    _get_xr_scale,
    _get_xr_translate,
    _get_xr_units,
    _XarrayMetadata,
    _XarrayProps,
)

# Days between the datetime64 epoch (1970-01-01) and 2013-01-01, the base
# timestamp used throughout the datetime coordinate tests. The translate
# expectations below are derived from this offset so they stay readable
# instead of appearing as magic numbers.
_EPOCH_DAYS = 15706
_EPOCH_HOURS = _EPOCH_DAYS * 24
_EPOCH_MINUTES = _EPOCH_HOURS * 60
_EPOCH_SECONDS = _EPOCH_MINUTES * 60
_EPOCH_MILLISECONDS = _EPOCH_SECONDS * 1000
_EPOCH_MICROSECONDS = _EPOCH_MILLISECONDS * 1000
_EPOCH_NANOSECONDS = _EPOCH_MICROSECONDS * 1000


def _dt(values):
    """Build a ``datetime64[ns]`` coordinate array from ISO 8601 strings."""
    return np.array(values, dtype='datetime64[ns]')


@pytest.fixture
def data_array_factory():
    """Factory for building DataArrays with a given shape, dims, and coords."""

    def _build(shape, dims, coords=None, attrs=None):
        return xr.DataArray(
            np.ones(shape), dims=dims, coords=coords or {}, attrs=attrs or {}
        )

    return _build


class TestCheckXarray:
    """_check_xarray: which xarray properties does the data expose?"""

    @pytest.mark.parametrize(
        ('data', 'expected'),
        [
            (
                xr.DataArray(np.ones((3, 4)), dims=['y', 'x']),
                _XarrayProps(has_dims=True, has_coords=True),
            ),
            (
                xr.Variable(('y', 'x'), np.ones((3, 4))),
                _XarrayProps(has_dims=True, has_coords=False),
            ),
            (
                np.ones((3, 4)),
                _XarrayProps(has_dims=False, has_coords=False),
            ),
            (
                [np.ones((5, 5)), np.ones((3, 3))],
                _XarrayProps(has_dims=False, has_coords=False),
            ),
        ],
        ids=['dataarray', 'variable', 'numpy', 'list'],
    )
    def test_props(self, data, expected):
        assert _check_xarray(data) == expected


class TestCoordMetadata:
    """_coord_metadata: per-axis scale/translate/unit from coordinate values."""

    @pytest.mark.parametrize(
        ('values', 'expected'),
        [
            # numeric
            ([10, 15, 20], _CoordMetadata(5.0, 10.0, None)),
            ([42], _CoordMetadata(1.0, 42.0, None)),
            ([88, 86, 84], _CoordMetadata(-2.0, 88.0, None)),
            (np.array([], dtype=float), _CoordMetadata(1.0, 0.0, None)),
            # string (categorical)
            (['a', 'b', 'c'], _CoordMetadata(1.0, 0.0, None)),
            # datetime64: missing (NaT) timestamps degrade to index space
            (
                _dt(['2013-01-01', 'NaT', '2013-01-03']),
                _CoordMetadata(1.0, 0.0, None),
            ),
            (
                _dt(['NaT', '2013-01-02', '2013-01-03']),
                _CoordMetadata(1.0, 0.0, None),
            ),
            # one entry per time unit, largest to smallest, plus the
            # nanosecond fallback for steps finer than any whole unit
            (
                _dt(['2013-01-01', '2013-01-04', '2013-01-07']),
                _CoordMetadata(3.0, float(_EPOCH_DAYS), 'day'),
            ),
            (
                _dt(
                    [
                        '2013-01-01',
                        '2013-01-01 06:00:00',
                        '2013-01-01 12:00:00',
                    ]
                ),
                _CoordMetadata(6.0, float(_EPOCH_HOURS), 'hour'),
            ),
            (
                _dt(
                    [
                        '2013-01-01',
                        '2013-01-01 00:30:00',
                        '2013-01-01 01:00:00',
                    ]
                ),
                _CoordMetadata(30.0, float(_EPOCH_MINUTES), 'minute'),
            ),
            (
                _dt(
                    [
                        '2013-01-01',
                        '2013-01-01 00:01:30',
                        '2013-01-01 00:03:00',
                    ]
                ),
                _CoordMetadata(90.0, float(_EPOCH_SECONDS), 'second'),
            ),
            (
                _dt(
                    [
                        '2013-01-01T00:00:00.000',
                        '2013-01-01T00:00:00.500',
                        '2013-01-01T00:00:01.000',
                    ]
                ),
                _CoordMetadata(
                    500.0, float(_EPOCH_MILLISECONDS), 'millisecond'
                ),
            ),
            (
                _dt(
                    [
                        '2013-01-01T00:00:00.000000',
                        '2013-01-01T00:00:00.000500',
                        '2013-01-01T00:00:01.000000',
                    ]
                ),
                _CoordMetadata(
                    500.0, float(_EPOCH_MICROSECONDS), 'microsecond'
                ),
            ),
            (
                _dt(
                    [
                        '2013-01-01T00:00:00.000000000',
                        '2013-01-01T00:00:00.000000500',
                        '2013-01-01T00:00:01.000000000',
                    ]
                ),
                _CoordMetadata(500.0, float(_EPOCH_NANOSECONDS), 'nanosecond'),
            ),
        ],
        ids=[
            'numeric',
            'single',
            'negative',
            'empty',
            'string',
            'nat-second',
            'nat-first',
            'day',
            'hour',
            'minute',
            'second',
            'millisecond',
            'microsecond',
            'nanosecond',
        ],
    )
    def test_coord_metadata(self, values, expected):
        assert _coord_metadata(np.asarray(values)) == expected


class TestGetXrScale:
    """_get_xr_scale: per-axis scale extracted from a DataArray's coords."""

    @pytest.mark.parametrize(
        ('coords', 'shape', 'expected'),
        [
            ({'y': [0, 5, 10], 'x': [0, 2, 4, 6]}, (3, 4), [5.0, 2.0]),
            (
                {'y': ('y', ['a', 'b', 'c']), 'x': [0, 2, 4, 6]},
                (3, 4),
                [1.0, 2.0],
            ),
            (
                {
                    'y': _dt(['2013-01-01', '2013-01-04', '2013-01-07']),
                    'x': [0, 2, 4, 6],
                },
                (3, 4),
                [3.0, 2.0],
            ),
        ],
        ids=['numeric', 'string', 'datetime'],
    )
    def test_scale(self, data_array_factory, coords, shape, expected):
        data_array = data_array_factory(shape, ['y', 'x'], coords)
        assert _get_xr_scale(data_array, data_array.dims) == expected


class TestGetXrTranslate:
    """_get_xr_translate: per-axis offset from a DataArray's coords."""

    @pytest.mark.parametrize(
        ('coords', 'shape', 'expected'),
        [
            (
                {'y': [10, 15, 20], 'x': [100, 102, 104, 106]},
                (3, 4),
                [10.0, 100.0],
            ),
            (
                {'y': ('y', ['a', 'b', 'c']), 'x': [0, 2, 4, 6]},
                (3, 4),
                [0.0, 0.0],
            ),
            (
                {
                    'y': _dt(['2013-01-01', '2013-01-04', '2013-01-07']),
                    'x': [0, 2, 4, 6],
                },
                (3, 4),
                [float(_EPOCH_DAYS), 0.0],
            ),
        ],
        ids=['numeric', 'string', 'datetime'],
    )
    def test_translate(self, data_array_factory, coords, shape, expected):
        data_array = data_array_factory(shape, ['y', 'x'], coords)
        assert _get_xr_translate(data_array, data_array.dims) == expected


class TestGetXrUnits:
    """_get_xr_units: per-axis units from coord attrs / datetime dtype."""

    @pytest.mark.parametrize(
        ('coords', 'expected'),
        [
            (
                {
                    'y': ('y', [0, 1, 2], {'units': 'microns'}),
                    'x': ('x', [0, 1, 2, 3], {'units': 'mm'}),
                },
                ['microns', 'mm'],
            ),
            (
                {
                    'y': ('y', [0, 1, 2], {'units': 'microns'}),
                    'x': [0, 1, 2, 3],  # no attrs
                },
                ['microns', None],
            ),
            (
                {
                    'y': ('y', [0, 1, 2], {'units': 'not_a_real_unit_ever'}),
                    'x': [0, 1, 2, 3],
                },
                [None, None],
            ),
            (
                {
                    'y': ('y', [0, 1, 2], {'units': 'microns'}),
                    'x': (
                        'x',
                        [0, 1, 2, 3],
                        {'units': 'not_a_real_unit_ever'},
                    ),
                },
                ['microns', None],
            ),
        ],
        ids=['all', 'partial', 'invalid', 'mixed'],
    )
    def test_from_attrs(self, data_array_factory, coords, expected):
        data_array = data_array_factory((3, 4), ['y', 'x'], coords)
        assert _get_xr_units(data_array, data_array.dims) == expected

    @pytest.mark.parametrize(
        ('values', 'expected_unit'),
        [
            (_dt(['2013-01-01', '2013-01-02', '2013-01-03']), 'day'),
            (
                _dt(
                    [
                        '2013-01-01',
                        '2013-01-01 06:00:00',
                        '2013-01-01 12:00:00',
                    ]
                ),
                'hour',
            ),
            (
                _dt(
                    [
                        '2013-01-01T00:00:00.000',
                        '2013-01-01T00:00:00.500',
                        '2013-01-01T00:00:01.000',
                    ]
                ),
                'millisecond',
            ),
            (
                _dt(
                    [
                        '2013-01-01T00:00:00.000000000',
                        '2013-01-01T00:00:00.000000500',
                        '2013-01-01T00:00:01.000000000',
                    ]
                ),
                'nanosecond',
            ),
        ],
        ids=['day', 'hour', 'millisecond', 'nanosecond'],
    )
    def test_datetime_derives_unit(
        self, data_array_factory, values, expected_unit
    ):
        """Datetime coords report the time unit derived from their spacing."""
        data_array = data_array_factory(
            (3, 4), ['y', 'x'], {'y': values, 'x': [0, 2, 4, 6]}
        )
        assert _get_xr_units(data_array, data_array.dims) == [
            expected_unit,
            None,
        ]

    def test_datetime_units_attr_ignored(self, data_array_factory):
        """A units attr on a datetime coord is ignored to match its scale."""
        data_array = data_array_factory(
            (3, 4),
            ['y', 'x'],
            {
                'y': (
                    'y',
                    _dt(
                        [
                            '2013-01-01',
                            '2013-01-01 06:00:00',
                            '2013-01-01 12:00:00',
                        ]
                    ),
                    {'units': 'days'},
                ),
                'x': [0, 2, 4, 6],
            },
        )
        # scale is 6 (hours); the unit must be 'hour', not the 'days' attr
        assert _get_xr_scale(data_array, data_array.dims) == [6.0, 2.0]
        assert _get_xr_units(data_array, data_array.dims) == ['hour', None]

    def test_datetime_nat_has_no_unit(self, data_array_factory):
        """NaT timestamps degrade to index space with no unit."""
        data_array = data_array_factory(
            (3, 4),
            ['y', 'x'],
            {'y': _dt(['2013-01-01', 'NaT', '2013-01-03']), 'x': [0, 2, 4, 6]},
        )
        assert _get_xr_units(data_array, data_array.dims) == [None, None]

    def test_registered_cf_aliases(self, monkeypatch, data_array_factory):
        """CF aliases registered in the pint registry are recognized.

        A private registry is used (via monkeypatch) so this test never
        mutates napari's global application registry, which would otherwise
        leak into other tests.
        """
        import pint

        ureg = pint.UnitRegistry()
        ureg.define('degrees_north = degree')
        ureg.define('degrees_east = degree')
        monkeypatch.setattr(pint, 'get_application_registry', lambda: ureg)

        data_array = data_array_factory(
            (3, 4),
            ['y', 'x'],
            {
                'y': ('y', [0, 1, 2], {'units': 'degrees_north'}),
                'x': ('x', [0, 1, 2, 3], {'units': 'degrees_east'}),
            },
        )
        assert _get_xr_units(data_array, data_array.dims) == [
            'degrees_north',
            'degrees_east',
        ]


class TestGetXrMetadata:
    """_get_xr_metadata: orchestration of label/scale/translate/units."""

    def test_non_xarray_noop(self):
        """Plain arrays are a no-op: no metadata is inferred."""
        assert _get_xr_metadata(np.ones((3, 4))) == _XarrayMetadata()

    def test_variable_only_labels(self):
        """Variable contributes axis_labels only (it has no coords)."""
        v = xr.Variable(('y', 'x'), np.ones((3, 4)))
        assert _get_xr_metadata(v) == _XarrayMetadata(
            axis_labels=('y', 'x'), scale=None, translate=None, units=None
        )

    def test_dataarray_infers_all_fields(self, data_array_factory):
        """DataArray contributes labels, scale, translate, and units."""
        data_array = data_array_factory(
            (3, 4),
            ['y', 'x'],
            {'y': ('y', [0, 5, 10], {'units': 'microns'}), 'x': [0, 2, 4, 6]},
        )
        assert _get_xr_metadata(data_array) == _XarrayMetadata(
            axis_labels=('y', 'x'),
            scale=[5.0, 2.0],
            translate=[0.0, 0.0],
            units=['microns', None],
        )

    def test_explicit_overrides_win(self, data_array_factory):
        """Explicitly provided metadata is never overwritten."""
        data_array = data_array_factory(
            (3, 4), ['y', 'x'], {'y': [0, 5, 10], 'x': [0, 2, 4, 6]}
        )
        meta = _get_xr_metadata(
            data_array,
            axis_labels=('row', 'col'),
            scale=[3.0, 3.0],
            translate=[1.0, 1.0],
            units=['mm', 'mm'],
        )
        assert meta == _XarrayMetadata(
            axis_labels=('row', 'col'),
            scale=[3.0, 3.0],
            translate=[1.0, 1.0],
            units=['mm', 'mm'],
        )

    def test_rgb_excludes_color_axis(self, data_array_factory):
        """rgb excludes the trailing color axis from all inferred fields."""
        data_array = data_array_factory(
            (3, 4, 3),
            ['y', 'x', 'channel'],
            {
                'y': ('y', [0, 5, 10], {'units': 'microns'}),
                'x': [0, 2, 4, 6],
                'channel': [0, 1, 2],
            },
        )
        assert _get_xr_metadata(data_array, rgb=True) == _XarrayMetadata(
            axis_labels=('y', 'x'),
            scale=[5.0, 2.0],
            translate=[0.0, 0.0],
            units=['microns', None],
        )

    def test_rgb_explicit_overrides_win(self, data_array_factory):
        """Explicit metadata still wins when rgb is set."""
        data_array = data_array_factory(
            (3, 4, 3),
            ['y', 'x', 'channel'],
            {'y': [0, 5, 10], 'x': [0, 2, 4, 6], 'channel': [0, 1, 2]},
        )
        meta = _get_xr_metadata(
            data_array,
            rgb=True,
            axis_labels=('row', 'col'),
            scale=[3.0, 3.0],
            translate=[1.0, 1.0],
            units=['mm', 'mm'],
        )
        assert meta == _XarrayMetadata(
            axis_labels=('row', 'col'),
            scale=[3.0, 3.0],
            translate=[1.0, 1.0],
            units=['mm', 'mm'],
        )
