from string import Formatter
from typing import Any, Sequence, Union

import numpy as np
from typing_extensions import Literal

from napari.utils.events.custom_types import Array
from napari.utils.translations import trans

from .style_encoding import (
    _ConstantStyleEncoding,
    _DerivedStyleEncoding,
    _ManualStyleEncoding,
)

"""A scalar array that represents one string value."""
StringValue = Array[str, ()]

"""An Nx1 array where each element represents one string value."""
StringArray = Array[str, (-1,)]


"""The default string to use, which may also be used a safe fallback string."""
DEFAULT_STRING = ''


class ConstantStringEncoding(_ConstantStyleEncoding[StringValue, StringArray]):
    """Encodes color values from a single constant color.

    Attributes
    ----------
    constant : StringValue
        The constant string value.
    encoding_type : Literal['ConstantStringEncoding']
        The type of encoding this specifies, which is useful for distinguishing
        this from other encodings when passing this as a dictionary.
    """

    constant: StringValue
    encoding_type: Literal['ConstantStringEncoding'] = 'ConstantStringEncoding'


class ManualStringEncoding(_ManualStyleEncoding[StringValue, StringArray]):
    """Encodes string values manually in an array.

    Attributes
    ----------
    array : StringArray
        The array of string values.
    default : StringValue
        The default string value that is used when requesting a value that
        is out of bounds in the array attribute.
    encoding_type : Literal['ManualStringEncoding']
        The type of encoding this specifies, which is useful for distinguishing
        this from other encodings when passing this as a dictionary.
    """

    array: StringArray = []
    default: StringValue = DEFAULT_STRING
    encoding_type: Literal['ManualStringEncoding'] = 'ManualStringEncoding'


class DirectStringEncoding(_DerivedStyleEncoding[StringValue, StringArray]):
    """Encodes strings directly from a feature column.

    Attributes
    ----------
    feature : str
        The name of the feature that contains the desired strings.
    fallback : StringValue
        The safe constant fallback string to use if the feature column
        does not contain valid string values.
    encoding_type : Literal['DirectStringEncoding']
        The type of encoding this specifies, which is useful for distinguishing
        this from other encodings when passing this as a dictionary.
    """

    feature: str
    fallback: StringValue = DEFAULT_STRING
    encoding_type: Literal['DirectStringEncoding'] = 'DirectStringEncoding'

    def __call__(self, features: Any) -> StringArray:
        return np.array(features[self.feature], dtype=str)


class FormatStringEncoding(_DerivedStyleEncoding[StringValue, StringArray]):
    """Encodes string values by formatting feature values.

    Attributes
    ----------
    format : str
        A format string with the syntax supported by :func:`str.format`,
        where all format fields should be feature names.
    fallback : StringValue
        The safe constant fallback string to use if the format string
        is not valid or contains fields other than feature names.
    encoding_type : Literal['FormatStringEncoding']
        The type of encoding this specifies, which is useful for distinguishing
        this from other encodings when passing this as a dictionary.
    """

    format: str
    fallback: StringValue = DEFAULT_STRING
    encoding_type: Literal['FormatStringEncoding'] = 'FormatStringEncoding'

    def __call__(self, features: Any) -> StringArray:
        values = features.apply(
            lambda row: self.format.format(**row),
            axis='columns',
            result_type='reduce',
        )
        return np.array(values, dtype=str)


# Define supported encodings as tuples instead of Union, so that they can be used with
# isinstance without relying on get_args, which was only added in python 3.8.

"""The string encodings supported by napari in order of precedence."""
_STRING_ENCODINGS = (
    FormatStringEncoding,
    DirectStringEncoding,
    ConstantStringEncoding,
    ManualStringEncoding,
)

StringEncodingUnion = Union[_STRING_ENCODINGS]

StringEncodingArgument = Union[
    StringEncodingUnion, dict, str, Sequence[str], None
]


def validate_string_encoding(
    value: StringEncodingArgument,
) -> StringEncodingUnion:
    """Validates and coerces an input to a StringEncoding.

    Parameters
    ----------
    value : StringEncodingArgument
        The value to validate and coerce.
        If this is already one of the supported string encodings, it is returned as is.
        If this is a dict, then it should represent one of the supported string encodings.
        If this a valid format string, then a FormatStringEncoding is returned.
        If this is any other string, a ConstantStringEncoding is returned.
        If this is a sequence of strings, a ManualStringEncoding is returned.

    Returns
    -------
    StringEncodingUnion
        An instance of one of the support string encodings.

    Raises
    ------
    TypeError
        If the input is not a supported type.
    ValidationError
        If the input cannot be parsed into a StringEncoding.
    """
    if value is None:
        return ConstantStringEncoding(constant=DEFAULT_STRING)
    if isinstance(value, _STRING_ENCODINGS):
        return value
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        if _is_format_string(value):
            return FormatStringEncoding(format=value)
        return ConstantStringEncoding(constant=value)
    if isinstance(value, Sequence):
        return ManualStringEncoding(array=value, default='')
    raise TypeError(
        trans._(
            'value should be one of the support string encodings, a dict, a string, a sequence of strings, or None',
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
    True if format contains at least one field, False otherwise.
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
