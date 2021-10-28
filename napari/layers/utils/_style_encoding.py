import warnings
from abc import ABC, abstractmethod
from enum import auto
from typing import Collection, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import ValidationError, parse_obj_as

from ...utils.events import EventedModel
from ...utils.misc import StringEnum


class EncodingType(StringEnum):
    """The encoding type, which is a constant field and useful for disambiguation of dict input."""

    CONSTANT = auto()
    DIRECT = auto()
    IDENTITY = auto()
    NOMINAL = auto()
    QUANTITATIVE = auto()
    FORMAT_STRING = auto()


class StyleEncoding(ABC):
    """Defines a way to encode style values, like colors and strings."""

    @abstractmethod
    def _get_array(
        self,
        properties: Dict[str, np.ndarray],
        n_rows: int,
        indices: Optional = None,
    ) -> np.ndarray:
        """Get the array of values generated from this and the given properties.

        If generating values from the given properties fails, this will fall back
        to returning some safe/default value.

        In general the returned value will be a read-only numpy array, as it may
        be a result from np.broadcast.

        Parameters
        ----------
        properties : Dict[str, np.ndarray]
            The properties from which to derive the output values.
        n_rows : int
            The total number of rows in the properties table.
        indices : Optional
            The row indices for which to return values. If None, return all of them.
            If not None, must be usable as indices for np.ndarray.

        Returns
        -------
        np.ndarray
            The numpy array of derived values. This is either a single value or
            has the same length as the given indices.
        """
        pass

    @abstractmethod
    def _clear(self):
        """Clears all previously generated values. Call this before _get_array to refresh values."""
        pass

    @abstractmethod
    def _append(self, array: np.ndarray):
        """Appends raw style values to this.

        This is useful for supporting the paste operation in layers.

        Parameters
        ----------
        array : np.ndarray
            The values to append. The dimensionality of these should match that of the existing style values.
        """
        pass

    @abstractmethod
    def _delete(self, indices):
        """Deletes style values from this by index.

        Parameters
        ----------
        indices
            The indices of the style values to remove. Must be usable as indices for np.ndarray.
        """
        pass


class ConstantStyleEncoding(EventedModel, StyleEncoding):
    """Encodes a constant style value.

    The _get_array method returns the constant broadcasted to the required length.

    Attributes
    ----------
    constant : np.ndarray
        The constant style value.
    """

    class Config:
        extra = 'forbid'

    constant: np.ndarray

    def _get_array(
        self,
        properties: Dict[str, np.ndarray],
        n_rows: int,
        indices: Optional[Collection[int]] = None,
    ) -> np.ndarray:
        output_length = n_rows if indices is None else len(indices)
        output_shape = (output_length,) + self.constant.shape
        return np.broadcast_to(self.constant, output_shape)

    def _append(self, array: np.ndarray):
        pass

    def _delete(self, indices):
        pass

    def _clear(self):
        pass


class DirectStyleEncoding(EventedModel, StyleEncoding):
    """Encodes style values directly.

    The style values are encoded directly in the array attribute, so that
    attribute can be written to make persistent updates.

    Attributes
    ----------
    array : np.ndarray
        The array of values.
    default : np.ndarray
        The default style value that is used when requesting a value that
        is out of bounds in the array attribute. In general this is a numpy
        array because color is a 1D RGBA numpy array, but mostly this will
        be a 0D numpy array (i.e. a scalar).
    """

    class Config:
        extra = 'forbid'

    array: np.ndarray
    default: np.ndarray

    def _get_array(
        self,
        properties: Dict[str, np.ndarray],
        n_rows: int,
        indices: Optional[Collection[int]] = None,
    ) -> np.ndarray:
        current_length = self.array.shape[0]
        if n_rows > current_length:
            tail_array = np.array([self.default] * (n_rows - current_length))
            self._append(tail_array)
        return self.array if indices is None else self.array[indices]

    def _append(self, array: np.ndarray):
        self.array = _append_maybe_empty(self.array, array)

    def _delete(self, indices):
        self.array = _delete_in_bounds(self.array, indices)

    def _clear(self):
        self.array = np.empty((0,))


class DerivedStyleEncoding(EventedModel, StyleEncoding, ABC):
    """Encodes style values by deriving them from property values."""

    class Config:
        extra = 'forbid'

    fallback: np.ndarray
    _array: np.ndarray

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._array = np.empty((0,))

    @abstractmethod
    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        pass

    def _get_array(
        self,
        properties: Dict[str, np.ndarray],
        n_rows: int,
        indices: Optional = None,
    ) -> np.ndarray:
        current_length = self._array.shape[0]
        tail_indices = range(current_length, n_rows)
        try:
            if len(tail_indices) > 0:
                tail_array = self._apply(properties, tail_indices)
                self._append(tail_array)
            return self._array if indices is None else self._array[indices]
        except (KeyError, ValueError) as error:
            self_str = repr(self)
            warnings.warn(
                '\n'
                'Applying the following derived encoding:\n'
                f'{self_str}\n'
                'failed with error:\n'
                f'{error}\n'
                f'Returning safe constant value instead.',
                category=RuntimeWarning,
            )
        return self.fallback

    def _append(self, array: np.ndarray):
        self._array = _append_maybe_empty(self._array, array)

    def _delete(self, indices):
        self._array = _delete_in_bounds(self._array, indices)

    def _clear(self):
        self._array = np.empty((0,))


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
        encoding_names = get_type_names(encodings)
        raise ValueError(
            'Original error:\n'
            f'{error}'
            'Failed to parse a supported encoding from kwargs:\n'
            f'{kwargs}\n\n'
            'The kwargs must specify the fields of exactly one of the following encodings:\n'
            f'{encoding_names}\n\n'
        )


def get_type_names(types: Iterable[type]) -> Tuple[str, ...]:
    """Gets the short names of the given types."""
    return tuple(type.__name__ for type in types)


def infer_n_rows(
    encoding: StyleEncoding, properties: Dict[str, np.ndarray]
) -> int:
    """Infers the number of rows in the given properties table."""
    if len(properties) > 0:
        return len(next(iter(properties)))
    if isinstance(encoding, DirectStyleEncoding):
        return len(encoding.array)
    return 1


def _delete_in_bounds(array: np.ndarray, indices) -> np.ndarray:
    # TODO: consider warning if any indices are OOB.
    safe_indices = [index for index in indices if index < array.shape[0]]
    return np.delete(array, safe_indices, axis=0)


def _append_maybe_empty(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if right.size == 0:
        return left
    if left.size == 0:
        return right
    return np.append(left, right, axis=0)
