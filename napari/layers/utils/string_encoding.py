from string import Formatter
from typing import Any, Dict, Sequence, Union

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
    IndicesType,
    StyleEncoding,
    parse_kwargs_as_encoding,
)

"""A scalar array that represents one string value."""
StringValue = Array[str, ()]

"""An Nx1 array where each element represents one string value."""
StringArray = Array[str, (-1,)]


@runtime_checkable
class StringEncoding(StyleEncoding[StringArray], Protocol):
    """Encodes strings from features."""


"""The default string to use, which may also be used a safe fallback string."""
DEFAULT_STRING = ''


class ConstantStringEncoding(ConstantStyleEncoding[StringValue, StringArray]):
    """Encodes color values from a single constant color.

    Attributes
    ----------
    constant : StringValue
        The constant string value.
    """

    encoding_type: EncodingType = Field(
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

    encoding_type: EncodingType = Field(
        EncodingType.DIRECT, const=EncodingType.DIRECT
    )
    array: StringArray = []
    default: StringValue = DEFAULT_STRING


class IdentityStringEncoding(DerivedStyleEncoding[StringValue, StringArray]):
    """Encodes strings directly from a feature column.

    Attributes
    ----------
    feature : str
        The name of the feature that contains the desired strings.
    fallback : StringValue
        The safe constant fallback string to use if the feature column
        does not contain valid string values.
    """

    encoding_type: EncodingType = Field(
        EncodingType.IDENTITY, const=EncodingType.IDENTITY
    )
    feature: str
    fallback: StringValue = DEFAULT_STRING

    def _apply(
        self,
        features: Any,
        indices: IndicesType,
    ) -> StringArray:
        return np.array(features[self.feature][indices], dtype=str)


class FormatStringEncoding(DerivedStyleEncoding[StringValue, StringArray]):
    """Encodes string values by formatting feature values.

    Attributes
    ----------
    format_string : str
        A format string with the syntax supported by :func:`str.format`,
        where all format fields should be feature names.
    fallback : StringValue
        The safe constant fallback string to use if the format string
        is not valid or contains fields other than feature names.
    """

    encoding_type: EncodingType = Field(
        EncodingType.FORMAT_STRING, const=EncodingType.FORMAT_STRING
    )
    format_string: str
    fallback: StringValue = DEFAULT_STRING

    def _apply(
        self,
        features: Any,
        indices: IndicesType,
    ) -> StringArray:
        # TODO: maybe exploit pandas API here.
        return np.array(
            [
                self.format_string.format(**_get_feature_row(features, index))
                for index in indices
            ]
        )


def _get_feature_row(features: Any, index: int) -> Dict[str, Any]:
    return {name: values[index] for name, values in features.items()}


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
    string: Union[StringEncoding, dict, str, Sequence[str], None],
) -> StringEncoding:
    """Validates and coerces an input to a StringEncoding.

    Parameters
    ----------
    string : Union[StringEncoding, dict, str, Sequence[str], None]
        The input or RHS of an assignment to a StringEncoding field. If this
        is already a StringEncoding, it is returned as is. If this is a dict,
        then we try to parse that as one of the built-in StringEncodings. If
        this is a valid
        Otherwise we try to parse the input as a direct encoding of multiple
        strings.

    Returns
    -------
    StringEncoding

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
        if _is_format_string(string):
            return FormatStringEncoding(format_string=string)
        return ConstantStringEncoding(constant=string)
    if isinstance(string, Sequence):
        return DirectStringEncoding(array=string, default='')
    raise TypeError(
        trans._(
            f'string should be one of {_STRING_ENCODING_NAMES}, a dict, str, iterable, or None',
            deferred=True,
        )
    )


def _is_format_string(string: str) -> bool:
    """Checks if a string is a valid format string with at least one field.

    Parameters
    ----------
    string : str
        The string to check.

    Returns
    -------
    True if format_string contains at least one field, False otherwise.
    """
    try:
        fields = tuple(
            field
            for _, field, _, _ in Formatter().parse(string)
            if field is not None
        )
    except ValueError:
        return False
    return len(fields) > 0
