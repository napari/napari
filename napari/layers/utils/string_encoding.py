from string import Formatter
from typing import Any, Sequence, Union

import numpy as np
from typing_extensions import Literal, Protocol, runtime_checkable

from napari.utils.events.custom_types import Array
from napari.utils.translations import trans

from .style_encoding import (
    EncodingType,
    StyleEncoding,
    _ConstantStyleEncoding,
    _DerivedStyleEncoding,
    _ManualStyleEncoding,
    parse_kwargs_as_encoding,
)

"""A scalar array that represents one string value."""
StringValue = Array[str, ()]

"""An Nx1 array where each element represents one string value."""
StringArray = Array[str, (-1,)]


@runtime_checkable
class StringEncoding(StyleEncoding[StringArray], Protocol):
    """Encodes strings from features.

    See ``validate_string_encoding`` for a description of the supported types
    that can be used when assigning a value to a StringEncoding field.
    """


def validate_string_encoding(
    string: Union[StringEncoding, dict, str, Sequence[str], None],
) -> StringEncoding:
    """Validates and coerces an input to a StringEncoding.

    Parameters
    ----------
    string : Union[StringEncoding, dict, str, Sequence[str], None]
        The value to validate and/or coerce.
        If this is already a StringEncoding, it is returned as is.
        If this is a dict, then it should represent one of the built-in StringEncodings.
        If this a valid format string, then a FormatStringEncoding is returned.
        If this is any other string, a ConstantStringEncoding is returned.
        If this is a sequence of strings, a ManualStringEncoding is returned.
        See the examples for some expected usage.

    Returns
    -------
    StringEncoding

    Raises
    ------
    TypeError
        If the input is not a supported type.
    ValidationError
        If the input cannot be parsed into a StringEncoding.

    Examples
    --------
    Leave an existing StringEncoding alone.
    >>> original = ConstantStringEncoding(constant='abc')
    >>> validated = validate_string_encoding(original)
    >>> id(original) == id(validated)
    True

    Coerce a dict to a DirectStringEncoding.
    >>> validate_string_encoding({'feature': 'class'})
    DirectStringEncoding(fallback=array('', dtype='<U1'), feature='class', encoding_type=<EncodingType.DIRECT: 'direct'>)

    Coerce a format string to a FormatStringEncoding.
    >>> validate_string_encoding('{class}: {score:.2f}')
    FormatStringEncoding(fallback=array('', dtype='<U1'), format_string='{class}: {score:.2f}', encoding_type=<EncodingType.FORMAT: 'format'>)

    Coerce a non-format string to a ConstantStringEncoding.
    >>> validate_string_encoding('abc')
    ConstantStringEncoding(constant=array('abc', dtype='<U3'), encoding_type=<EncodingType.CONSTANT: 'constant'>)

    Coerce a sequence of strings to a ManualStringEncoding.
    >>> validate_string_encoding(a', 'b', 'c'])
    ManualStringEncoding(array=array(['a', 'b', 'c'], dtype='<U1'), default=array('', dtype='<U1'), encoding_type=<EncodingType.MANUAL: 'manual'>)
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
        return ManualStringEncoding(array=string, default='')
    raise TypeError(
        trans._(
            'string should be a StringEncoding, dict, str, Sequence[str], or None',
            deferred=True,
        )
    )


"""The default string to use, which may also be used a safe fallback string."""
DEFAULT_STRING = ''


class ConstantStringEncoding(_ConstantStyleEncoding[StringValue, StringArray]):
    """Encodes color values from a single constant color.

    Attributes
    ----------
    constant : StringValue
        The constant string value.
    encoding_type : Literal['constant']
        The type of encoding this specifies, which is useful for distinguishing
        this from other encodings when passing this as a dictionary.
    """

    constant: StringValue
    encoding_type: Literal[EncodingType.CONSTANT] = 'constant'


class ManualStringEncoding(_ManualStyleEncoding[StringValue, StringArray]):
    """Encodes string values manually in an array.

    Attributes
    ----------
    array : StringArray
        The array of string values.
    default : StringValue
        The default string value that is used when requesting a value that
        is out of bounds in the array attribute.
    encoding_type : Literal['manual']
        The type of encoding this specifies, which is useful for distinguishing
        this from other encodings when passing this as a dictionary.
    """

    array: StringArray = []
    default: StringValue = DEFAULT_STRING
    encoding_type: Literal[EncodingType.MANUAL] = 'manual'


class DirectStringEncoding(_DerivedStyleEncoding[StringValue, StringArray]):
    """Encodes strings directly from a feature column.

    Attributes
    ----------
    feature : str
        The name of the feature that contains the desired strings.
    fallback : StringValue
        The safe constant fallback string to use if the feature column
        does not contain valid string values.
    encoding_type : Literal['direct']
        The type of encoding this specifies, which is useful for distinguishing
        this from other encodings when passing this as a dictionary.
    """

    feature: str
    fallback: StringValue = DEFAULT_STRING
    encoding_type: Literal[EncodingType.DIRECT] = 'direct'

    def __call__(self, features: Any) -> StringArray:
        return np.array(features[self.feature], dtype=str)


class FormatStringEncoding(_DerivedStyleEncoding[StringValue, StringArray]):
    """Encodes string values by formatting feature values.

    Attributes
    ----------
    format_string : str
        A format string with the syntax supported by :func:`str.format`,
        where all format fields should be feature names.
    fallback : StringValue
        The safe constant fallback string to use if the format string
        is not valid or contains fields other than feature names.
    encoding_type : Literal['format']
        The type of encoding this specifies, which is useful for distinguishing
        this from other encodings when passing this as a dictionary.
    """

    format_string: str
    fallback: StringValue = DEFAULT_STRING
    encoding_type: Literal[EncodingType.FORMAT] = 'format'

    def __call__(self, features: Any) -> StringArray:
        values = features.apply(
            lambda row: self.format_string.format(**row), axis='columns'
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
