from abc import ABC, abstractmethod
from string import Formatter
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import numpy as np
from pydantic import Field

from napari.utils.events.custom_types import Array
from napari.utils.translations import trans

from ._style_encoding import (
    ConstantStyleEncoding,
    DerivedStyleEncoding,
    DirectStyleEncoding,
    EncodingType,
    get_type_names,
    parse_kwargs_as_encoding,
)

"""A scalar array that represents one string value."""
StringArray = Array[str, ()]

"""An Nx1 array where each element represents one string value."""
MultiStringArray = Array[str, (-1,)]


class StringEncoding(ABC):
    @abstractmethod
    def _get_array(
        self,
        properties: Dict[str, np.ndarray],
        n_rows: int,
        indices: Optional = None,
    ) -> MultiStringArray:
        pass

    @abstractmethod
    def _clear(self):
        pass

    @abstractmethod
    def _append(self, array: MultiStringArray):
        pass

    @abstractmethod
    def _delete(self, indices):
        pass


"""The default string to use, which may also be used a safe fallback string."""
DEFAULT_STRING = ''


class ConstantStringEncoding(ConstantStyleEncoding, StringEncoding):
    """Encodes color values from a single constant color.

    Attributes
    ----------
    constant : StringArray
        The constant string value.
    """

    type: EncodingType = Field(
        EncodingType.CONSTANT, const=EncodingType.CONSTANT
    )
    constant: StringArray


class DirectStringEncoding(DirectStyleEncoding, StringEncoding):
    """Encodes string values directly in an array.

    Attributes
    ----------
    array : MultiStringArray
        The array of string values.
    default : StringArray
        The default string value that is used when requesting a value that
        is out of bounds in the array attribute.
    """

    type: EncodingType = Field(EncodingType.DIRECT, const=EncodingType.DIRECT)
    array: MultiStringArray = []
    default: StringArray = DEFAULT_STRING


class IdentityStringEncoding(DerivedStyleEncoding, StringEncoding):
    """Encodes strings directly from a property column.

    Attributes
    ----------
    property : str
        The name of the property that contains the desired strings.
    fallback : StringArray
        The safe constant fallback string to use if the property column
        does not contain valid string values.
    """

    type: EncodingType = Field(
        EncodingType.IDENTITY, const=EncodingType.IDENTITY
    )
    property: str
    fallback: StringArray = DEFAULT_STRING

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        return np.array(properties[self.property][indices], dtype=str)


class FormatStringEncoding(DerivedStyleEncoding, StringEncoding):
    """Encodes string values by formatting property values.

    Attributes
    ----------
    format_string : str
        A format string with the syntax supported by :func:`str.format`,
        where all format fields should be property names.
    fallback : StringArray
        The safe constant fallback string to use if the format string
        is not valid or contains fields other than property names.
    """

    type: EncodingType = Field(
        EncodingType.FORMAT_STRING, const=EncodingType.FORMAT_STRING
    )
    format_string: str
    fallback: StringArray = DEFAULT_STRING

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
_STRING_ENCODINGS = (
    FormatStringEncoding,
    IdentityStringEncoding,
    ConstantStringEncoding,
    DirectStringEncoding,
)

_STRING_ENCODING_NAMES = get_type_names(_STRING_ENCODINGS)


def validate_string_encoding(
    string: Union[StringEncoding, dict, str, Iterable[str], None],
    properties: Dict[str, np.ndarray],
) -> StringEncoding:
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
    if isinstance(string, Sequence):
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
