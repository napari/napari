import numpy as np
import xarray as xr

from napari.utils._xarray_utils import (
    _check_xarray,
    _get_xr_axis_labels,
    _get_xr_metadata,
    _get_xr_scale,
    _get_xr_translate,
    _get_xr_units,
    _XarrayMetadata,
    _XarrayProps,
)


class TestCheckXarray:
    def test_dataarray(self):
        """DataArray has both dims and coords."""
        da = xr.DataArray(np.ones((3, 4)), dims=['y', 'x'])
        caps = _check_xarray(da)
        assert caps == _XarrayProps(has_dims=True, has_coords=True)

    def test_variable(self):
        """Variable (NamedArray) has dims but not coords."""
        v = xr.Variable(('y', 'x'), np.ones((3, 4)))
        caps = _check_xarray(v)
        assert caps == _XarrayProps(has_dims=True, has_coords=False)

    def test_numpy(self):
        """Numpy array has neither dims nor coords."""
        caps = _check_xarray(np.ones((3, 4)))
        assert caps == _XarrayProps(False, False)

    def test_list(self):
        """List (multiscale) has neither dims nor coords."""
        caps = _check_xarray([np.ones((5, 5)), np.ones((3, 3))])
        assert caps == _XarrayProps(False, False)


class TestGetXrAxisLabels:
    def test_dataarray(self):
        """Axis labels extracted from DataArray dims."""
        da = xr.DataArray(np.ones((3, 4)), dims=['y', 'x'])
        assert _get_xr_axis_labels(da) == ('y', 'x')

    def test_variable(self):
        """Axis labels extracted from Variable (has dims, no coords)."""
        v = xr.Variable(('y', 'x'), np.ones((3, 4)))
        assert _get_xr_axis_labels(v) == ('y', 'x')


class TestGetXrScale:
    def test_normal(self):
        """Scale computed from equally-spaced coordinate values."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={'y': [0, 5, 10], 'x': [0, 2, 4, 6]},
        )
        assert _get_xr_scale(da) == [5.0, 2.0]

    def test_single_element(self):
        """Single-element coords fall back to 1.0."""
        da = xr.DataArray(
            np.ones((1, 4)),
            dims=['y', 'x'],
            coords={'y': [0], 'x': [0, 2, 4, 6]},
        )
        assert _get_xr_scale(da) == [1.0, 2.0]

    def test_negative_spacing(self):
        """Decreasing coordinate values produce negative scale."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={'y': [88, 86, 84], 'x': [0, 2, 4, 6]},
        )
        assert _get_xr_scale(da) == [-2.0, 2.0]

    def test_datetime_coords(self):
        """Datetime coords use the largest whole time unit for scale."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={
                'y': np.array(
                    ['2000-01-01', '2000-01-04', '2000-01-07'],
                    dtype='datetime64[ns]',
                ),
                'x': [0, 2, 4, 6],
            },
        )
        # 3-day spacing -> day/3
        assert _get_xr_scale(da) == [3.0, 2.0]

    def test_sub_second_datetime_coords(self):
        """Sub-second datetime steps use ms/us/ns units."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={
                'y': np.array(
                    [
                        '2013-01-01T00:00:00.000',
                        '2013-01-01T00:00:00.500',
                        '2013-01-01T00:00:01.000',
                    ],
                    dtype='datetime64[ns]',
                ),
                'x': [0, 2, 4, 6],
            },
        )
        assert _get_xr_scale(da) == [500.0, 2.0]
        assert _get_xr_units(da) == ['millisecond', None]

    def test_nanosecond_fallback(self):
        """Steps finer than a microsecond fall back to nanoseconds."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={
                'y': np.array(
                    [
                        '2013-01-01T00:00:00.000000000',
                        '2013-01-01T00:00:00.000000500',
                        '2013-01-01T00:00:00.000001000',
                    ],
                    dtype='datetime64[ns]',
                ),
                'x': [0, 2, 4, 6],
            },
        )
        assert _get_xr_scale(da) == [500.0, 2.0]
        assert _get_xr_units(da) == ['nanosecond', None]

    def test_string_coords(self):
        """String (categorical) coords fall back to 1.0."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={'y': ('y', ['a', 'b', 'c']), 'x': [0, 2, 4, 6]},
        )
        assert _get_xr_scale(da) == [1.0, 2.0]

    def test_nat_coords(self):
        """Missing (NaT) timestamps degrade to index space, not crash."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={
                'y': np.array(
                    ['2013-01-01', 'NaT', '2013-01-03'],
                    dtype='datetime64[ns]',
                ),
                'x': [0, 2, 4, 6],
            },
        )
        assert _get_xr_scale(da) == [1.0, 2.0]
        assert _get_xr_units(da) == [None, None]


class TestGetXrTranslate:
    def test_normal(self):
        """Translate from first coordinate values."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={'y': [10, 15, 20], 'x': [100, 102, 104, 106]},
        )
        assert _get_xr_translate(da) == [10.0, 100.0]

    def test_single_element(self):
        """Single-element coords still return their first value."""
        da = xr.DataArray(
            np.ones((1, 4)),
            dims=['y', 'x'],
            coords={'y': [42], 'x': [0, 2, 4, 6]},
        )
        assert _get_xr_translate(da) == [42.0, 0.0]

    def test_datetime_coords(self):
        """Datetime coords translate is since the datetime64 epoch."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={
                'y': np.array(
                    ['2000-01-01', '2000-01-04', '2000-01-07'],
                    dtype='datetime64[ns]',
                ),
                'x': [0, 2, 4, 6],
            },
        )
        # 2000-01-01 is day 10957 since the 1970-01-01 epoch
        assert _get_xr_translate(da) == [10957.0, 0.0]

    def test_sub_second_datetime_coords(self):
        """Sub-second datetime translate is in the derived unit."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={
                'y': np.array(
                    [
                        '2013-01-01T00:00:00.000',
                        '2013-01-01T00:00:00.500',
                        '2013-01-01T00:00:01.000',
                    ],
                    dtype='datetime64[ns]',
                ),
                'x': [0, 2, 4, 6],
            },
        )
        # 2013-01-01T00:00:00 in milliseconds since the 1970-01-01 epoch
        assert _get_xr_translate(da) == [1356998400000.0, 0.0]

    def test_nanosecond_fallback(self):
        """Sub-microsecond datetime translate is in nanoseconds."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={
                'y': np.array(
                    [
                        '2013-01-01T00:00:00.000000000',
                        '2013-01-01T00:00:00.000000500',
                        '2013-01-01T00:00:01.000000000',
                    ],
                    dtype='datetime64[ns]',
                ),
                'x': [0, 2, 4, 6],
            },
        )
        # 2013-01-01T00:00:00 in nanoseconds since the 1970-01-01 epoch
        assert _get_xr_translate(da) == [1356998400000000000.0, 0.0]

    def test_string_coords(self):
        """String (categorical) coords fall back to 0.0."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={'y': ('y', ['a', 'b', 'c']), 'x': [0, 2, 4, 6]},
        )
        assert _get_xr_translate(da) == [0.0, 0.0]


class TestGetXrUnits:
    def test_all_present(self):
        """Units read from all coord attrs."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={
                'y': ('y', [0, 1, 2], {'units': 'microns'}),
                'x': ('x', [0, 1, 2, 3], {'units': 'mm'}),
            },
        )
        assert _get_xr_units(da) == ['microns', 'mm']

    def test_partial(self):
        """Missing units attrs yield None for those axes."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={
                'y': ('y', [0, 1, 2], {'units': 'microns'}),
                'x': [0, 1, 2, 3],  # no attrs
            },
        )
        assert _get_xr_units(da) == ['microns', None]

    def test_invalid_unit_filtered(self):
        """Unrecognized unit strings are silently dropped."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={
                'y': ('y', [0, 1, 2], {'units': 'not_a_real_unit_ever'}),
            },
        )
        assert _get_xr_units(da) == [None, None]

    def test_mixed_valid_invalid(self):
        """Valid and invalid units mixed."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={
                'y': ('y', [0, 1, 2], {'units': 'microns'}),
                'x': ('x', [0, 1, 2, 3], {'units': 'not_a_real_unit_ever'}),
            },
        )
        assert _get_xr_units(da) == ['microns', None]

    def test_datetime_derives_unit(self):
        """Datetime coords with no units attr report the derived time unit."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={
                'y': np.array(
                    ['2013-01-01', '2013-01-02', '2013-01-03'],
                    dtype='datetime64[ns]',
                ),
                'x': [0, 1, 2, 3],
            },
        )
        assert _get_xr_units(da) == ['day', None]

    def test_datetime_units_attr_ignored(self):
        """A units attr on a datetime coord is ignored to stay consistent."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={
                'y': (
                    'y',
                    np.array(
                        [
                            '2013-01-01',
                            '2013-01-01 06:00:00',
                            '2013-01-01 12:00:00',
                        ],
                        dtype='datetime64[ns]',
                    ),
                    {'units': 'days'},
                ),
                'x': [0, 2, 4, 6],
            },
        )
        # scale is 6 (hours); units must report 'hour', not the 'days' attr
        assert _get_xr_scale(da) == [6.0, 2.0]
        assert _get_xr_units(da) == ['hour', None]

    def test_registered_cf_aliases(self):
        """CF aliases registered in the pint registry are recognized."""
        import pint

        ureg = pint.get_application_registry()
        if 'degrees_north' not in ureg:
            ureg.define('degrees_north = degree')
        if 'degrees_east' not in ureg:
            ureg.define('degrees_east = degree')
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={
                'y': ('y', [0, 1, 2], {'units': 'degrees_north'}),
                'x': ('x', [0, 1, 2, 3], {'units': 'degrees_east'}),
            },
        )
        assert _get_xr_units(da) == ['degrees_north', 'degrees_east']


class TestGetXrMetadata:
    def test_variable_only_labels(self):
        """Variable only contributes axis_labels (no coords)."""
        v = xr.Variable(('y', 'x'), np.ones((3, 4)))
        assert _get_xr_metadata(v) == _XarrayMetadata(
            axis_labels=('y', 'x'),
            scale=None,
            translate=None,
            units=None,
        )

    def test_dataarray_all_fields(self):
        """DataArray contributes labels, scale, translate, units."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={
                'y': ('y', [0, 5, 10], {'units': 'microns'}),
                'x': [0, 2, 4, 6],
            },
        )
        meta = _get_xr_metadata(da)
        assert meta.axis_labels == ('y', 'x')
        assert meta.scale == [5.0, 2.0]
        assert meta.translate == [0.0, 0.0]
        assert meta.units == ['microns', None]

    def test_explicit_overrides_win(self):
        """Explicitly provided metadata is not overwritten."""
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={'y': [0, 5, 10], 'x': [0, 2, 4, 6]},
        )
        meta = _get_xr_metadata(
            da,
            axis_labels=('row', 'col'),
            scale=[3.0, 3.0],
            units=['mm', 'mm'],
        )
        assert meta.axis_labels == ('row', 'col')
        assert meta.scale == [3.0, 3.0]
        # translate was not passed, so it is still inferred
        assert meta.translate == [0.0, 0.0]
        assert meta.units == ['mm', 'mm']
