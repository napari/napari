import numpy as np
import pytest
import xarray as xr

from napari.utils._xarray_utils import (
    _check_xarray,
    _get_xr_axis_labels,
    _get_xr_scale,
    _get_xr_translate,
    _get_xr_units,
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

    def test_cf_units_when_available(self):
        """CF unit names are accepted when cf_xarray is installed."""
        pytest.importorskip('cf_xarray.units')
        da = xr.DataArray(
            np.ones((3, 4)),
            dims=['y', 'x'],
            coords={
                'y': ('y', [0, 1, 2], {'units': 'degrees_north'}),
                'x': ('x', [0, 1, 2, 3], {'units': 'degrees_east'}),
            },
        )
        units = _get_xr_units(da)
        assert units == ['degrees_north', 'degrees_east']
