from abc import ABC, abstractmethod
from string import Formatter
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import Field, ValidationError, parse_obj_as, validator

from ...utils.events import EventedModel


class StyleEncoding(EventedModel, ABC):
    """Defines a way to encode style values, like colors and strings."""

    _array: np.ndarray = np.empty((0,))

    @abstractmethod
    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        pass

    def _get_array(
        self,
        properties: Dict[str, np.ndarray],
        n_rows: int,
        indices: Sequence[int],
    ) -> np.ndarray:
        """Get the array of values generated from this and the given properties.

        Parameters
        ----------
        properties : Dict[str, np.ndarray]
            The properties from which to derive the output values.
        n_rows : int
            The total number of rows in the properties table.
        indices : Sequence[int]
            The row indices for which to return values.

        Returns
        -------
        np.ndarray
            The numpy array of derived values. This is either a single value or
            has the same length as the given indices.

        Raises
        ------
        ValueError
            If this is not compatible with the given properties. In this case, you
            should probably fall back to a safer encoding (e.g. a constant).
        """
        current_length = self._array.shape[0]
        tail_indices = range(current_length, n_rows)
        tail_array = self._apply(properties, tail_indices)
        self._append(tail_array)
        return self._array[indices]

    def _clear(self):
        """Clear the stored values. Call this before _get_array to refresh values."""
        self._array = np.empty((0,))

    def _append(self, array: np.ndarray):
        """Appends raw style values to this.

        This is useful for supporting the paste operation in layers.

        Parameters
        ----------
        array : np.ndarray
            The values to append. The dimensionality of these should match that of the existing style values.
        """
        self._array = _append_maybe_empty(self._array, array)

    def _delete(self, indices: Iterable[int]):
        """Deletes style values from this by index.

        Parameters
        ----------
        indices : Iterable[int]
            The indices of the style values to remove.
        """
        self._array = np.delete(self._array, list(indices), axis=0)


class DirectStyleEncoding(StyleEncoding):
    """Encodes style values directly.

    The style values are encoded directly in the array attribute, so that
    attribute can be written to make persistent updates.

    Attributes
    ----------
    default : np.ndarray
        The default style value that is used when requesting a value that
        is out of bounds in the array attribute. In general this is a numpy
        array because color is a 1D RGBA numpy array, but mostly this will
        be a 0D numpy array (i.e. a scalar).
    """

    default: np.ndarray

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        n_values = self.array.shape[0]
        in_bound_indices = [index for index in indices if index < n_values]
        n_default = len(indices) - len(in_bound_indices)
        return _append_maybe_empty(
            self.array[in_bound_indices],
            np.array([self.default] * n_default),
        )


class ConstantStringEncoding(StyleEncoding):
    """Encodes string values directly in an array.

    Attributes
    ----------
    constant : str
        The string that is always returned regardless of property values.
    """

    constant: str

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        return np.repeat(self.constant, len(indices))


class IdentityStringEncoding(StyleEncoding):
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

    def _validate_properties(self, properties: Dict[str, np.ndarray]):
        _check_property_name(properties, self.property_name)


class FormatStringEncoding(StyleEncoding):
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

    def _validate_properties(self, properties: Dict[str, np.ndarray]):
        is_format_string(properties, self.format_string)


class DirectStringEncoding(DirectStyleEncoding):
    """Encodes string values directly in an array."""

    @validator('array', pre=True, always=True)
    def _check_array(cls, array) -> np.ndarray:
        return np.array(array, dtype=str)

    @validator('default', pre=True, always=True)
    def _check_default(cls, default) -> np.ndarray:
        return np.array(default, dtype=str)


def parse_kwargs_as_encoding(encodings: Tuple[type, ...], **kwargs):
    """Parses the given kwargs as one of the given encodings.

    Parameters
    ----------
    encodings : Tuple[type, ...]
        The supported encoding types, each of which must be a subclass of
        :class:`StyleEncoding`. The first encoding that can be constructed
        from the given kwargs will be returned.

    Raises
    ------
    ValueError
        If the provided kwargs cannot be used to construct any of the given encodings.
    """
    try:
        return parse_obj_as(Union[encodings], kwargs)
    except ValidationError as error:
        raise ValueError(
            'Original error:\n'
            f'{error}'
            'Failed to parse a supported encoding from kwargs:\n'
            f'{kwargs}\n\n'
            'The kwargs must specify the fields of exactly one of the following encodings:\n'
            f'{encodings}\n\n'
        )


def is_format_string(
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


# Define supported encodings as tuples instead of Union, so that they can be used with
# isinstance without relying on get_args, which was only added in python 3.8.
# The order of the encoding matters because we rely on Pydantic for parsing dict kwargs
# inputs for each encoding - it will use the first encoding that can be defined with those kwargs.

"""The string encodings supported by napari in order of precedence."""
STRING_ENCODINGS = (
    FormatStringEncoding,
    IdentityStringEncoding,
    ConstantStringEncoding,
    DirectStringEncoding,
)


def _append_maybe_empty(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if right.size == 0:
        return left
    if left.size == 0:
        return right
    return np.append(left, right, axis=0)


def _calculate_contrast_limits(
    values: np.ndarray,
) -> Optional[Tuple[float, float]]:
    contrast_limits = None
    if values.size > 0:
        min_value = np.min(values)
        max_value = np.max(values)
        # Use < instead of != to handle nans.
        if min_value < max_value:
            contrast_limits = (min_value, max_value)
    return contrast_limits


def _get_property_row(
    properties: Dict[str, np.ndarray], index: int
) -> Dict[str, Any]:
    return {name: values[index] for name, values in properties.items()}


def _check_property_name(
    properties: Dict[str, np.ndarray], property_name: str
):
    if property_name not in properties:
        raise ValueError(f'{property_name} is not in properties: {properties}')
