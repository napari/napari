"""Utilities for extracting napari layer metadata from xarray DataArrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import ArrayLike


__all__ = (
    '_CoordMetadata',
    '_XarrayMetadata',
    '_XarrayProps',
    '_check_xarray',
    '_coord_metadata',
    '_data_dims',
    '_datetime_metadata',
    '_get_xr_metadata',
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


class _XarrayMetadata(NamedTuple):
    """Inferred layer metadata.

    ``None`` fields mean no value could be inferred for that metadata type.
    """

    axis_labels: tuple[str, ...] | None = None
    scale: list[float] | None = None
    translate: list[float] | None = None
    units: list[str | None] | None = None


class _CoordMetadata(NamedTuple):
    """Scale, translate, and unit inferred for one dimension's coordinate.

    ``unit`` is ``None`` for numeric or string coordinates (napari then uses
    its default units) and set to the derived time unit for
    ``datetime64`` coordinates.
    """

    scale: float
    translate: float
    unit: str | None


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


def _data_dims(
    data: xr.DataArray | xr.Variable,
    rgb: bool = False,
) -> tuple[str, ...]:
    """Return the data's dims, possibly excluding the trailing RGB axis."""
    dims = list(data.dims)
    if rgb:
        dims.pop()
    return tuple(dims)


# time units, from largest to smallest, mapped to their length in
# nanoseconds.
_TIME_UNITS: dict[str, int] = {
    'day': 86_400 * 10**9,
    'hour': 3_600 * 10**9,
    'minute': 60 * 10**9,
    'second': 10**9,
    'millisecond': 10**6,
    'microsecond': 10**3,
}


def _datetime_metadata(
    values: np.ndarray,
) -> _CoordMetadata | None:
    """Return metadata for a ``datetime64`` coordinate, or ``None``.

    Picks the largest whole time unit (from day down to nanosecond) that
    divides the spacing between the first two values, so a monthly axis
    yields ``scale=31, unit='day'``, a 6-hourly axis ``scale=6,
    unit='hour'``, and a 500 ms axis ``scale=500, unit='millisecond'``.
    ``translate`` is the first value expressed in that unit, anchored at
    the ``datetime64`` epoch (1970-01-01) so that layers with different
    time resolutions share a consistent world frame.
    """
    if values.size < 2 or not np.issubdtype(values.dtype, np.datetime64):
        return None
    if np.isnat(values[0]) or np.isnat(values[1]):
        # missing (NaT) timestamps give no meaningful spacing/offset; the
        # caller falls back to index/pixel space
        return None
    # integer nanoseconds keeps the divisibility check exact: float modulo
    # is unreliable for sub-second units (e.g. 0.5 % 0.001 != 0).
    step_ns = int((values[1] - values[0]) / np.timedelta64(1, 'ns'))
    translate_ns = int(
        (
            values[0].astype('datetime64[ns]')
            - np.datetime64('1970-01-01', 'ns')
        )
        / np.timedelta64(1, 'ns')
    )
    for unit, ns in _TIME_UNITS.items():
        if abs(step_ns) % ns == 0:
            return _CoordMetadata(
                scale=float(step_ns) / ns,
                translate=translate_ns / ns,
                unit=unit,
            )
    # anything finer than a microsecond falls back to nanoseconds (the
    # resolution of datetime64)
    return _CoordMetadata(
        scale=float(step_ns),
        translate=float(translate_ns),
        unit='nanosecond',
    )


def _coord_metadata(
    values: np.ndarray,
) -> _CoordMetadata:
    """Return metadata for one dimension's coordinate.

    Numeric coordinates give the spacing between the first two values and
    the first value; ``datetime64`` coordinates give real-time units via
    :func:`_datetime_metadata`; single-element, empty, or string
    coordinates fall back to index/pixel space (scale 1.0, translate 0.0,
    unit ``None``).  ``unit`` is only set for ``datetime64`` coordinates
    (``attrs['units']`` is handled separately by :func:`_get_xr_units`).
    """
    dt = _datetime_metadata(values)
    if dt is not None:
        return dt
    if values.size > 0 and np.issubdtype(values.dtype, np.number):
        scale = float(values[1] - values[0]) if values.size >= 2 else 1.0
        return _CoordMetadata(
            scale=scale, translate=float(values[0]), unit=None
        )
    return _CoordMetadata(scale=1.0, translate=0.0, unit=None)


def _get_xr_scale(
    data: xr.DataArray,
    dims: tuple[str, ...],
) -> list[float]:
    """Infer scale from coordinate spacing for the given dims.

    Numeric coordinates give the spacing between the first two values;
    ``datetime64`` coordinates give real-time units (see
    :func:`_datetime_metadata`); single-element or string coordinates fall
    back to 1.0 (index/pixel space).
    """
    return [_coord_metadata(data.coords[d].values).scale for d in dims]


def _get_xr_translate(
    data: xr.DataArray,
    dims: tuple[str, ...],
) -> list[float]:
    """Infer translate (offset) from coordinates for the given dims.

    Numeric coordinates give the first value; ``datetime64`` coordinates
    give the offset in the derived time unit, anchored at the ``datetime64``
    epoch; string coordinates fall back to 0.0.
    """
    return [_coord_metadata(data.coords[d].values).translate for d in dims]


def _get_xr_units(
    data: xr.DataArray,
    dims: tuple[str, ...],
) -> list[str | None]:
    """Read units from coordinate attrs, validating against pint.

    Uses the CF convention (``coord.attrs['units']``).  Strings that pint
    cannot parse (e.g. ``'degrees_north'`` — CF-compliant but not a unit
    pint knows on its own) are silently dropped and replaced with ``None``,
    so napari uses its default (pixel) for that axis.  ``datetime64``
    coordinates with no usable ``units`` attr instead report the time unit
    derived by :func:`_get_xr_scale` (day down to nanosecond).

    Users can register additional unit conventions with pint so napari can
    recognise them, e.g.::

        import pint
        ureg = pint.get_application_registry()
        ureg.define('degrees_north = degree')

    napari deliberately does not bundle CF unit conventions itself.
    """
    from napari.utils.transforms._units import get_unit_from_name

    units: list[str | None] = []
    for dim in dims:
        coord = data.coords[dim]
        if np.issubdtype(coord.values.dtype, np.datetime64):
            # scale/translate are derived from the datetime spacing, so the
            # unit must match; an explicit units attr could disagree and is
            # ignored for datetime coordinates
            units.append(_coord_metadata(coord.values).unit)
            continue
        unit = coord.attrs.get('units')
        if unit is not None:
            try:
                get_unit_from_name(unit)
            except ValueError:
                unit = None
        units.append(unit)
    return units


def _get_xr_metadata(
    data: ArrayLike,
    *,
    rgb: bool = False,
    axis_labels: tuple[str, ...] | None = None,
    scale: list[float] | None = None,
    translate: list[float] | None = None,
    units: list[str | None] | None = None,
) -> _XarrayMetadata:
    """Return layer metadata inherited from *data*.

    Initially checks that *data* is Xarray-like (Variable or DataArray) and if
    not is a no-op returning original values.

    Any field passed in as ``None`` is inferred from *data* where possible;
    explicitly provided values pass through unchanged.

    rgb : bool, optional
        If True, the trailing axis is treated as the RGB/RGBA color axis and
        excluded from the inferred metadata. This is *not* equivalent to
        `channel_axis` because that is sliced out before layer init.

    Note: Only ``axis_labels``s inferred for ``Variable``-like objects,
    which have no coordinates.
    """
    props = _check_xarray(data)
    if props.has_dims:
        data_xr = cast('xr.DataArray | xr.Variable', data)
        dims = _data_dims(data_xr, rgb)
        if axis_labels is None:
            axis_labels = dims
        if props.has_coords:
            data_array = cast('xr.DataArray', data)
            if scale is None:
                scale = _get_xr_scale(data_array, dims)
            if translate is None:
                translate = _get_xr_translate(data_array, dims)
            if units is None:
                units = _get_xr_units(data_array, dims)
    return _XarrayMetadata(axis_labels, scale, translate, units)
