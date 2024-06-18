from __future__ import annotations

from collections.abc import Sequence
from typing import (
    Optional,
    Union,
    overload,
)

import pint

UnitsLike = Union[None, str, pint.Unit, dict[str, Union[str, pint.Unit]]]
UnitsInfo = Union[None, pint.Unit, dict[str, pint.Unit]]


__all__ = (
    'coerce_units_and_axes',
    'get_units_from_name',
    'UnitsLike',
    'UnitsInfo',
)


def coerce_units_and_axes(
    units: UnitsLike, axis_labels: Optional[Sequence[str]]
) -> tuple[UnitsInfo, Optional[list[str]]]:
    units_ = get_units_from_name(units)
    if axis_labels is None:
        return units_, None

    axis_labels = list(axis_labels)
    if len(axis_labels) != len(set(axis_labels)):
        raise ValueError('Axis labels must be unique.')

    if isinstance(units_, dict):
        if set(axis_labels).issubset(set(units_)):
            units_ = {name: units_[name] for name in axis_labels}
        else:
            diff = ', '.join(set(axis_labels) - set(units_))
            raise ValueError(
                'If both axis_labels and units are provided, '
                'all axis_labels must have a corresponding unit. '
                f'Missing units for: {diff}'
            )
    return units_, axis_labels


@overload
def get_units_from_name(units: None) -> None: ...


@overload
def get_units_from_name(units: Union[str, pint.Unit]) -> pint.Unit: ...


@overload
def get_units_from_name(
    units: dict[str, Union[str, pint.Unit]]
) -> dict[str, pint.Unit]: ...


def get_units_from_name(units: UnitsLike) -> UnitsInfo:
    """
    Convert a string or dict of strings to unyt units.
    """
    try:
        if isinstance(units, str):
            return pint.get_application_registry()[units].units
        if isinstance(units, dict):
            return {
                name: (
                    value
                    if isinstance(value, pint.Unit)
                    else pint.get_application_registry()[value].units
                )
                for name, value in units.items()
            }
    except AttributeError as e:
        raise ValueError(f'Could not find unit {units}') from e
    return units
