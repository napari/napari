from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from typing import (
    Union,
    overload,
)

import pint

UnitsLike = Union[None, str, pint.Unit, Sequence[str | pint.Unit]]
UnitsInfo = Union[None, pint.Unit, tuple[pint.Unit, ...]]


__all__ = (
    'UnitsInfo',
    'UnitsLike',
    'get_units_from_name',
)


def get_unit_from_name(unit: str | pint.Unit | None) -> pint.Unit:
    if isinstance(unit, pint.Unit):
        return unit
    if unit is None:
        return pint.get_application_registry().pixel
    if isinstance(unit, str):
        with suppress(pint.errors.UndefinedUnitError):
            return pint.get_application_registry().parse_expression(unit).units

    raise ValueError(f'Could not find unit {unit}')


@overload
def get_units_from_name(units: None) -> None: ...


@overload
def get_units_from_name(units: str | pint.Unit | None) -> pint.Unit: ...


@overload
def get_units_from_name(
    units: Sequence[str | pint.Unit | None],
) -> tuple[pint.Unit, ...]: ...


def get_units_from_name(units: UnitsLike) -> UnitsInfo:
    """Convert a string or sequence of strings to pint units."""
    if isinstance(units, str) or units is None:
        return get_unit_from_name(units)
    if isinstance(units, Sequence):
        return tuple(get_unit_from_name(u) for u in units)
    raise ValueError(f'Could not find unit {units}')
