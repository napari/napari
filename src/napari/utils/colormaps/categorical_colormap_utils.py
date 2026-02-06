import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from napari.utils.translations import trans


class Cycle(itertools.cycle):
    """A picklable cycle object.

    Python 3.14 removed pickling from itertools, but this workaround
    seems to work ok even for deepcopy of numpy arrays; however,
    This should ideally not be stored as a `cycle` object anymore.
    """

    def __copy__(self):
        return Cycle(self)

    def __deepcopy__(self, memo):
        return Cycle(self)


@dataclass(eq=False)
class ColorCycle:
    """A dataclass to hold a color cycle for the fallback_colors
    in the CategoricalColormap

    Attributes
    ----------
    values : np.ndarray
        The (Nx4) color array of all colors contained in the color cycle.
    cycle : Cycle
        The cycle object that gives fallback colors.
    """

    values: np.ndarray
    cycle: Cycle

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        validate = core_schema.no_info_plain_validator_function(
            cls.validate_type
        )
        json_serialize = core_schema.plain_serializer_function_ser_schema(
            lambda v: {
                'values': v.values.tolist()
                if isinstance(v.values, np.ndarray)
                else v
            },
            when_used='json',
        )
        return core_schema.json_or_python_schema(
            json_schema=validate,
            python_schema=validate,
            serialization=json_serialize,
        )

    @classmethod
    def validate_type(cls, val):
        # turn a generic dict into object
        if isinstance(val, dict):
            return _coerce_colorcycle_from_dict(val)
        if isinstance(val, ColorCycle):
            return val

        return _coerce_colorcycle_from_colors(val)

    def __eq__(self, other):
        if isinstance(other, ColorCycle):
            eq = np.array_equal(self.values, other.values)
        else:
            eq = False
        return eq


def _coerce_colorcycle_from_dict(
    val: dict[str, str | list | np.ndarray | Cycle],
) -> ColorCycle:
    # avoid circular import
    from napari.layers.utils.color_transformations import (
        transform_color,
        transform_color_cycle,
    )

    # validate values
    color_values = val.get('values')
    if color_values is None:
        raise ValueError(
            trans._('ColorCycle requires a values argument', deferred=True)
        )

    transformed_color_values = transform_color(color_values)

    # validate cycle
    color_cycle = val.get('cycle')
    if color_cycle is None:
        transformed_color_cycle = transform_color_cycle(
            color_cycle=color_values,
            elem_name='color_cycle',
            default='white',
        )[0]
    elif isinstance(color_cycle, Cycle):
        transformed_color_cycle = color_cycle
    elif isinstance(color_cycle, Iterable):
        # Workaround for https://github.com/pydantic/pydantic/issues/8907
        color_cycle = Cycle(color_cycle)
        transformed_color_cycle = color_cycle
    else:
        raise TypeError(
            f'cycle entry must be of type Cycle, got {type(color_cycle)}'
        )

    return ColorCycle(
        values=transformed_color_values, cycle=transformed_color_cycle
    )


def _coerce_colorcycle_from_colors(
    val: str | list | np.ndarray,
) -> ColorCycle:
    # avoid circular import
    from napari.layers.utils.color_transformations import (
        transform_color_cycle,
    )

    if isinstance(val, str):
        val = [val]
    (
        transformed_color_cycle,
        transformed_color_values,
    ) = transform_color_cycle(
        color_cycle=val,
        elem_name='color_cycle',
        default='white',
    )
    return ColorCycle(
        values=transformed_color_values, cycle=transformed_color_cycle
    )


def compare_colormap_dicts(cmap_1, cmap_2):
    if len(cmap_1) != len(cmap_2):
        return False
    for k, v in cmap_1.items():
        if k not in cmap_2:
            return False
        if not np.allclose(v, cmap_2[k]):
            return False
    return True
