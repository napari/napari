import warnings
from abc import abstractmethod
from enum import auto
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np
from pydantic import ValidationError, parse_obj_as
from typing_extensions import Protocol, runtime_checkable

from ...utils.events import EventedModel
from ...utils.misc import StringEnum

IndicesType = Union[List[int], np.ndarray]
StyleValue = TypeVar('StyleValue', bound=np.ndarray)
StyleArray = TypeVar('StyleArray', bound=np.ndarray)


class EncodingType(StringEnum):
    """The encoding type, which is a constant field and useful for disambiguation of dict input."""

    CONSTANT = auto()
    DIRECT = auto()
    IDENTITY = auto()
    NOMINAL = auto()
    QUANTITATIVE = auto()
    FORMAT_STRING = auto()


@runtime_checkable
class StyleEncoding(Protocol[StyleArray]):
    """Defines a way to encode style values, like colors and strings."""

    def _get_array(
        self,
        properties: Dict[str, np.ndarray],
        n_rows: int,
        indices: Optional[IndicesType] = None,
    ) -> StyleArray:
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
        StyleArray
            The numpy array of derived values. This has same length as the given indices
            and in general is read-only.
        """
        pass

    def _clear(self):
        """Clears all previously generated values. Call this before _get_array to refresh values."""
        pass

    def _append(self, array: StyleArray):
        """Appends raw style values to this.

        This is useful for supporting the paste operation in layers.

        Parameters
        ----------
        array : StyleArray
            The values to append. The dimensionality of these should match that of the existing style values.
        """
        pass

    def _delete(self, indices):
        """Deletes style values from this by index.

        Parameters
        ----------
        indices
            The indices of the style values to remove. Must be usable as indices for np.ndarray.
        """
        pass


class StyleEncodingModel(EventedModel, Generic[StyleValue, StyleArray]):
    class Config:
        extra = 'forbid'


class ConstantStyleEncoding(StyleEncodingModel[StyleValue, StyleArray]):
    """Encodes a constant style value.

    The _get_array method returns the constant broadcast to the required length.

    Attributes
    ----------
    constant : StyleValue
        The constant style value.
    """

    constant: StyleValue

    def _get_array(
        self,
        properties: Dict[str, np.ndarray],
        n_rows: int,
        indices: Optional[IndicesType] = None,
    ) -> StyleArray:
        return _broadcast_constant(self.constant, n_rows, indices)

    def _append(self, array: StyleArray):
        pass

    def _delete(self, indices):
        pass

    def _clear(self):
        pass


class DirectStyleEncoding(StyleEncodingModel[StyleValue, StyleArray]):
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

    array: StyleArray
    default: StyleValue

    def _get_array(
        self,
        properties: Dict[str, np.ndarray],
        n_rows: int,
        indices: Optional[IndicesType] = None,
    ) -> StyleArray:
        current_length = self.array.shape[0]
        if n_rows > current_length:
            tail_array = np.array([self.default] * (n_rows - current_length))
            self._append(tail_array)
        return _maybe_index_array(self.array, indices)

    def _append(self, array: StyleArray):
        self.array = np.append(self.array, array, axis=0)

    def _delete(self, indices):
        self.array = _delete_in_bounds(self.array, indices)

    def _clear(self):
        self.array = _empty_like_multi_array(self.default)


class DerivedStyleEncoding(StyleEncodingModel[StyleValue, StyleArray]):
    """Encodes style values by deriving them from property values."""

    fallback: StyleValue
    _array: StyleArray

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._clear()

    @abstractmethod
    def _apply(self, properties: Dict[str, np.ndarray], indices) -> StyleArray:
        pass

    def _get_array(
        self,
        properties: Dict[str, np.ndarray],
        n_rows: int,
        indices: Optional[IndicesType] = None,
    ) -> StyleArray:
        current_length = self._array.shape[0]
        tail_indices = range(current_length, n_rows)
        try:
            if len(tail_indices) > 0:
                tail_array = self._apply(properties, tail_indices)
                self._append(tail_array)
            return _maybe_index_array(self._array, indices)
        except (KeyError, ValueError) as error:
            self_str = repr(self)
            warnings.warn(
                '\n'
                'Applying the following derived encoding:\n'
                f'{self_str}\n'
                'failed with error:\n'
                f'{error}\n'
                f'Returning safe fallback value instead.',
                category=RuntimeWarning,
            )
        return _broadcast_constant(self.fallback, n_rows, indices)

    def _append(self, array: StyleArray):
        self._array = np.append(self._array, array, axis=0)

    def _delete(self, indices):
        self._array = _delete_in_bounds(self._array, indices)

    def _clear(self):
        self._array = _empty_like_multi_array(self.fallback)


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
        encoding_names = tuple(enc.__name__ for enc in encodings)
        raise ValueError(
            'Original error:\n'
            f'{error}'
            'Failed to parse a supported encoding from kwargs:\n'
            f'{kwargs}\n\n'
            'The kwargs must specify the fields of exactly one of the following encodings:\n'
            f'{encoding_names}\n\n'
        )


def _empty_like_multi_array(single_array: np.ndarray):
    shape = (0,) + single_array.shape
    return np.empty_like(single_array, shape=shape)


def _delete_in_bounds(array: np.ndarray, indices) -> np.ndarray:
    # TODO: do we really need bounds checking here?
    safe_indices = [i for i in indices if i < array.shape[0]]
    return np.delete(array, safe_indices, axis=0)


def _broadcast_constant(
    constant: np.ndarray, n_rows: int, indices: Optional[IndicesType]
):
    output_length = n_rows if indices is None else len(indices)
    output_shape = (output_length,) + constant.shape
    return np.broadcast_to(constant, output_shape)


def _maybe_index_array(array: np.ndarray, indices: Optional[IndicesType]):
    return array if indices is None else array[indices]
