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
    units: UnitsLike, axes_labels: Optional[Sequence[str]]
) -> tuple[UnitsInfo, Optional[list[str]]]:
    units_ = get_units_from_name(units)
    if axes_labels is None:
        return units_, None

    axes_labels = list(axes_labels)
    if len(axes_labels) != len(set(axes_labels)):
        raise ValueError('Axes labels must be unique.')

    if isinstance(units_, dict):
        if set(axes_labels).issubset(set(units_)):
            units_ = {name: units_[name] for name in axes_labels}
        else:
            diff = ', '.join(set(axes_labels) - set(units_))
            raise ValueError(
                'If both axes_labels and units are provided, '
                'all axes_labels must have a corresponding unit. '
                f'Missing units for: {diff}'
            )
    return units_, axes_labels


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
