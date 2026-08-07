"""Utilities for extracting napari layer metadata from xarray DataArrays."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import ArrayLike


__all__ = (
    '_XarrayProps',
    '_check_xarray',
    '_get_xr_axis_labels',
    '_get_xr_scale',
    '_get_xr_translate',
    '_get_xr_units',
)


class _XarrayProps(NamedTuple):
    """Properties of an xarray-like data object.

    Attributes
    ----------
    has_dims : bool
        True if data exposes ``.dims`` (Variable / DataArray).
    has_coords : bool
        True if data exposes ``.coords`` (DataArray only).
    """

    has_dims: bool = False
    has_coords: bool = False


def _check_xarray(data: ArrayLike) -> _XarrayProps:
    """Check what xarray properties *data* exposes.

    Returns a named tuple with ``has_dims`` (True for Variable / DataArray)
    and ``has_coords`` (True for DataArray only).
    """
    # xarray is an optional dependency, so it may not be importable.
    try:
        import xarray as xr
    except ImportError:
        return _XarrayProps()

    if isinstance(data, xr.DataArray):
        return _XarrayProps(has_dims=True, has_coords=True)
    if isinstance(data, xr.Variable):
        return _XarrayProps(has_dims=True)

    return _XarrayProps()


def _get_xr_axis_labels(data: xr.DataArray | xr.Variable) -> tuple[str, ...]:
    """Infer axis labels from xarray dims."""
    return tuple(str(d) for d in data.dims)


def _get_xr_scale(data: xr.DataArray) -> list[float]:
    """Infer scale from first coordinate values spacing.

    Assumes coordinates are linearly spaced. Falls back to 1.0 for
    single-element dimensions (size < 2), where spacing is undefined.
    """
    return [
        float(data.coords[d].values[1] - data.coords[d].values[0])
        if data.coords[d].size >= 2
        else 1.0
        for d in data.dims
    ]


def _get_xr_translate(data: xr.DataArray) -> list[float]:
    """Infer translate (offset) from the first coordinate value.

    Returns ``coord.values[0]`` for each dimension, which is the
    physical offset of the first pixel along that axis.
    """
    return [float(data.coords[d].values[0]) for d in data.dims]


def _get_xr_units(data: xr.DataArray) -> list[str | None]:
    """Read units from coordinate attrs, validating against pint.

    Uses the CF convention (``coord.attrs['units']``).  Invalid unit
    strings (e.g. ``'degrees_north'``, CF-compliant but not valid in
    pint alone) are silently dropped and replaced with ``None``, so
    napari will use its default (pixel) for that axis.

    If ``cf_xarray`` is installed its unit registry is loaded
    automatically, making many CF-standard unit names valid.
    """
    # Optionally register CF conventions with pint so that strings
    # like 'degrees_north', 'degrees_east', 'days since ...' etc.
    # are recognised as valid pint units.
    with suppress(ImportError):
        import cf_xarray.units  # noqa: F401

    from napari.utils.transforms._units import get_unit_from_name

    units: list[str | None] = []
    for dim in data.dims:
        unit = data.coords[dim].attrs.get('units')
        if unit is not None:
            try:
                get_unit_from_name(unit)
            except ValueError:
                unit = None
        units.append(unit)
    return units
