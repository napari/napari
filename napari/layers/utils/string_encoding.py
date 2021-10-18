from abc import ABC
from string import Formatter
from typing import Any, Dict, Iterable, Sequence, Union

import numpy as np
from pydantic import Field, validator

from napari.layers.utils.style_encoding import (
    ConstantStyleEncoding,
    DerivedStyleEncoding,
    DirectStyleEncoding,
    parse_kwargs_as_encoding,
)
from napari.utils.translations import trans

DEFAULT_STRING = np.array('')


class ConstantStringEncoding(ConstantStyleEncoding):
    """Encodes a constant string value."""

    @validator('constant', pre=True, always=True)
    def _validate_constant(cls, constant):
        return np.array(constant, dtype=str)


class DirectStringEncoding(DirectStyleEncoding):
    """Encodes string values directly in an array."""

    @validator('array', pre=True, always=True)
    def _validate_array(cls, array) -> np.ndarray:
        return np.array(array, dtype=str)

    @validator('default', pre=True, always=True)
    def _validate_default(cls, default) -> np.ndarray:
        return np.array(default, dtype=str)


class DerivedStringEncoding(DerivedStyleEncoding, ABC):
    def _fallback_value(self) -> np.ndarray:
        return DEFAULT_STRING


class IdentityStringEncoding(DerivedStringEncoding):
    """Encodes strings directly from a property column.

    Attributes
    ----------
    property_name : str
        The name of the property that contains the desired strings.
    """

    property_name: str = Field(..., allow_mutation=False)

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        return np.array(properties[self.property_name][indices], dtype=str)


class FormatStringEncoding(DerivedStringEncoding):
    """Encodes string values by formatting property values.

    Attributes
    ----------
    format_string : str
        A format string with the syntax supported by :func:`str.format`,
        where all format fields should be property names.
    """

    format_string: str = Field(..., allow_mutation=False)

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        return np.array(
            [
                self.format_string.format(
                    **_get_property_row(properties, index)
                )
                for index in indices
            ]
        )


# Define supported encodings as tuples instead of Union, so that they can be used with
# isinstance without relying on get_args, which was only added in python 3.8.

"""The string encodings supported by napari in order of precedence."""
STRING_ENCODINGS = (
    FormatStringEncoding,
    IdentityStringEncoding,
    ConstantStringEncoding,
    DirectStringEncoding,
)


def parse_string_encoding(
    string: Union[Union[STRING_ENCODINGS], dict, str, Iterable[str], None],
    properties: Dict[str, np.ndarray],
) -> Union[STRING_ENCODINGS]:
    if string is None:
        return ConstantStringEncoding(constant=DEFAULT_STRING)
    if isinstance(string, STRING_ENCODINGS):
        return string
    if isinstance(string, dict):
        return parse_kwargs_as_encoding(STRING_ENCODINGS, **string)
    if isinstance(string, str):
        if string in properties:
            return IdentityStringEncoding(property_name=string)
        if _is_format_string(properties, string):
            return FormatStringEncoding(format_string=string)
        return ConstantStringEncoding(constant=string)
    if isinstance(string, Sequence):
        return DirectStringEncoding(array=string, default='')
    raise TypeError(
        trans._(
            'string should be a StringEncoding, dict, str, iterable, or None',
            deferred=True,
        )
    )


def _is_format_string(
    properties: Dict[str, np.ndarray], format_string: str
) -> bool:
    """Returns true if the given string should be used in :class:`StringFormatEncoding`.

    Parameters
    ----------
    properties : Dict[str, np.ndarray]
        The properties of a layer.
    format_string : str
        The format string.

    Returns
    -------
    True if format_string contains at least one field, False otherwise.

    Raises
    ------
    ValueError
        If the format_string is not valid (e.g. mismatching braces), or one of the
        format fields is not a property name.
    """
    fields = tuple(
        field
        for _, field, _, _ in Formatter().parse(format_string)
        if field is not None
    )
    for field in fields:
        if field not in properties:
            raise ValueError(
                f'Found format string field {field} without a corresponding property'
            )
    return len(fields) > 0


def _get_property_row(
    properties: Dict[str, np.ndarray], index: int
) -> Dict[str, Any]:
    return {name: values[index] for name, values in properties.items()}
