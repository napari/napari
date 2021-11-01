from string import Formatter
from typing import Any, Dict, Iterable, Union

import numpy as np
from pydantic import Field
from typing_extensions import Protocol, runtime_checkable

from napari.utils.events.custom_types import Array
from napari.utils.translations import trans

from .style_encoding import (
    ConstantStyleEncoding,
    DerivedStyleEncoding,
    DirectStyleEncoding,
    EncodingType,
    StyleEncoding,
    parse_kwargs_as_encoding,
)

"""A scalar array that represents one string value."""
StringValue = Array[str, ()]

"""An Nx1 array where each element represents one string value."""
StringArray = Array[str, (-1,)]


@runtime_checkable
class StringEncoding(StyleEncoding[StringArray], Protocol):
    """Encodes strings from properties."""


"""The default string to use, which may also be used a safe fallback string."""
DEFAULT_STRING = ''


class ConstantStringEncoding(ConstantStyleEncoding[StringValue, StringArray]):
    """Encodes color values from a single constant color.

    Attributes
    ----------
    constant : StringValue
        The constant string value.
    """

    type: EncodingType = Field(
        EncodingType.CONSTANT, const=EncodingType.CONSTANT
    )
    constant: StringValue


class DirectStringEncoding(DirectStyleEncoding[StringValue, StringArray]):
    """Encodes string values directly in an array.

    Attributes
    ----------
    array : StringArray
        The array of string values.
    default : StringValue
        The default string value that is used when requesting a value that
        is out of bounds in the array attribute.
    """

    type: EncodingType = Field(EncodingType.DIRECT, const=EncodingType.DIRECT)
    array: StringArray = []
    default: StringValue = DEFAULT_STRING


class IdentityStringEncoding(DerivedStyleEncoding[StringValue, StringArray]):
    """Encodes strings directly from a property column.

    Attributes
    ----------
    property : str
        The name of the property that contains the desired strings.
    fallback : StringValue
        The safe constant fallback string to use if the property column
        does not contain valid string values.
    """

    type: EncodingType = Field(
        EncodingType.IDENTITY, const=EncodingType.IDENTITY
    )
    property: str
    fallback: StringValue = DEFAULT_STRING

    def _apply(
        self, properties: Dict[str, np.ndarray], indices
    ) -> StringArray:
        return np.array(properties[self.property][indices], dtype=str)


class FormatStringEncoding(DerivedStyleEncoding[StringValue, StringArray]):
    """Encodes string values by formatting property values.

    Attributes
    ----------
    format_string : str
        A format string with the syntax supported by :func:`str.format`,
        where all format fields should be property names.
    fallback : StringValue
        The safe constant fallback string to use if the format string
        is not valid or contains fields other than property names.
    """

    type: EncodingType = Field(
        EncodingType.FORMAT_STRING, const=EncodingType.FORMAT_STRING
    )
    format_string: str
    fallback: StringValue = DEFAULT_STRING

    def _apply(
        self, properties: Dict[str, np.ndarray], indices
    ) -> StringArray:
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
_STRING_ENCODINGS = (
    FormatStringEncoding,
    IdentityStringEncoding,
    ConstantStringEncoding,
    DirectStringEncoding,
)

_STRING_ENCODING_NAMES = tuple(enc.__name__ for enc in _STRING_ENCODINGS)


def validate_string_encoding(
    string: Union[StringEncoding, dict, str, Iterable[str], None],
    properties: Dict[str, np.ndarray],
) -> StringEncoding:
    """Validates and coerces an input to a StringEncoding.

    Parameters
    ----------
    string : Union[StringEncoding, dict, str, Iterable[str], None]
        The input or RHS of an assignment to a StringEncoding field. If this
        is already a StringEncoding, it is returned as is. If this is a dict,
        then we try to parse that as one of the built-in StringEncodings. If
        this is a string and a property name, then we return an identity
        string encoding based on that property. If this is a string, but not
        a property name, we try to parse the input as a format string.
        Otherwise we try to parse the input as a direct encoding of multiple
        strings.
    properties : Dict[str, np.ndarray]
        The property values, which typically come from a layer.

    Returns
    -------
    ColorEncoding

    Raises
    ------
    TypeError
        If the input is not a supported type.
    ValidationError
        If the input cannot be parsed into a string encoding.
    """
    if string is None:
        return ConstantStringEncoding(constant=DEFAULT_STRING)
    if isinstance(string, StringEncoding):
        return string
    if isinstance(string, dict):
        return parse_kwargs_as_encoding(_STRING_ENCODINGS, **string)
    if isinstance(string, str):
        if string in properties:
            return IdentityStringEncoding(property=string)
        if _is_format_string(properties, string):
            return FormatStringEncoding(format_string=string)
        return ConstantStringEncoding(constant=string)
    if isinstance(string, Iterable):
        return DirectStringEncoding(array=string, default='')
    raise TypeError(
        trans._(
            f'string should be one of {_STRING_ENCODING_NAMES}, a dict, str, iterable, or None',
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
