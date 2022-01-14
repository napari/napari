import warnings
from abc import abstractmethod
from enum import auto
from typing import Any, Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np
from pydantic import ValidationError, parse_obj_as
from typing_extensions import Protocol, runtime_checkable

from ...utils.events import EventedModel
from ...utils.misc import StringEnum

IndicesType = Union[range, List[int], np.ndarray]

"""The variable type of a single style value."""
StyleValue = TypeVar('StyleValue', bound=np.ndarray)

"""The variable type of multiple style values in an array."""
StyleArray = TypeVar('StyleArray', bound=np.ndarray)


class EncodingType(StringEnum):
    """The encoding type, which is a constant field and useful for disambiguation of dict input."""

    CONSTANT = auto()
    MANUAL = auto()
    DIRECT = auto()
    NOMINAL = auto()
    QUANTITATIVE = auto()
    FORMAT_STRING = auto()


@runtime_checkable
class StyleEncoding(Protocol[StyleArray]):
    """Defines a way to encode style values, like colors and strings."""

    def __call__(
        self,
        features: Any,
        *,
        indices: Optional[IndicesType] = None,
    ) -> StyleArray:
        """Get the array of values generated from this and the given features.

        If generating values from the given features fails, this will fall back
        to returning some safe/default value.

        In general the returned value will be a read-only numpy array, as it may
        be a result from np.broadcast.

        Parameters
        ----------
        features : Dataframe-like
            The features from which to derive the output values.
        indices : Optional[IndicesType]
            The row indices for which to return values. If None, return all of them.

        Returns
        -------
        StyleArray
            The numpy array of derived values. This has same length as the given indices
            and in general is read-only.
        """

    def _apply(
        self,
        features: Any,
        indices: IndicesType,
    ) -> StyleArray:
        """Applies this encoding to the given features at the given row indices."""

    def _clear(self) -> None:
        """Clears all previously generated values.

        Call this before _get_array to refresh values.
        """

    def _append(self, array: StyleArray) -> None:
        """Appends raw style values to this.

        This is useful for supporting the paste operation in layers.

        Parameters
        ----------
        array : StyleArray
            The values to append. The dimensionality of these should match that of the existing style values.
        """

    def _delete(self, indices: IndicesType) -> None:
        """Deletes style values from this by index.

        Parameters
        ----------
        indices
            The indices of the style values to remove.
        """
        pass

    def _json_encode(self) -> dict:
        """Converts the encoding to a dict that should be convertible to JSON."""
        pass


class StyleEncodingModel(EventedModel, Generic[StyleValue, StyleArray]):
    class Config:
        # Forbid extra fields to ensure different types of encodings can be properly resolved.
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

    def __call__(
        self,
        features: Any,
        *,
        indices: Optional[IndicesType] = None,
    ) -> StyleArray:
        return _broadcast_constant(self.constant, features.shape[0], indices)

    def _apply(
        self,
        features: Any,
        indices: IndicesType,
    ) -> StyleArray:
        return _broadcast_constant(self.constant, len(indices), indices)

    def _append(self, array: StyleArray) -> None:
        pass

    def _delete(self, indices: IndicesType) -> None:
        pass

    def _clear(self) -> None:
        pass

    def _json_encode(self) -> dict:
        return self.dict()


class ManualStyleEncoding(StyleEncodingModel[StyleValue, StyleArray]):
    """Encodes style values manually.

    The style values are encoded manually in the array attribute, so that
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

    def __call__(
        self,
        features: Any,
        *,
        indices: Optional[IndicesType] = None,
    ) -> StyleArray:
        current_length = self.array.shape[0]
        n_rows = features.shape[0]
        if n_rows > current_length:
            tail_array = np.array([self.default] * (n_rows - current_length))
            self._append(tail_array)
        return _maybe_index_array(self.array, indices)

    def _apply(
        self,
        features: Any,
        indices: IndicesType,
    ) -> StyleArray:
        return np.array([self.default] * len(indices))

    def _append(self, array: StyleArray) -> None:
        self.array = np.append(self.array, array, axis=0)

    def _delete(self, indices: IndicesType) -> None:
        self.array = _delete_in_bounds(self.array, indices)

    def _clear(self) -> None:
        self.array = _empty_array_like(self.default)

    def _json_encode(self) -> dict:
        return self.dict()


class DerivedStyleEncoding(StyleEncodingModel[StyleValue, StyleArray]):
    """Encodes style values by deriving them from feature values."""

    fallback: StyleValue
    _array: StyleArray

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._array = _empty_array_like(self.fallback)

    @abstractmethod
    def _apply(self, features: Any, indices: IndicesType) -> StyleArray:
        pass

    def __call__(
        self,
        features: Any,
        *,
        indices: Optional[IndicesType] = None,
    ) -> StyleArray:
        current_length = self._array.shape[0]
        n_rows = features.shape[0]
        tail_indices = range(current_length, n_rows)
        try:
            if len(tail_indices) > 0:
                tail_array = self._apply(features, tail_indices)
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

    def _append(self, array: StyleArray) -> None:
        self._array = np.append(self._array, array, axis=0)

    def _delete(self, indices: IndicesType) -> None:
        self._array = _delete_in_bounds(self._array, indices)

    def _clear(self) -> None:
        self._array = _empty_array_like(self.fallback)

    def _json_encode(self) -> dict:
        return self.dict()


def parse_kwargs_as_encoding(encodings: Tuple[type, ...], **kwargs) -> Any:
    """Parses the given kwargs as one of the given encodings.

    Parameters
    ----------
    encodings : Tuple[type, ...]
        The supported encoding types, each of which must be a subclass of
        :class:`StyleEncoding`. The first encoding that can be constructed
        from the given kwargs will be returned.
    kwargs
        The keyword arguments of the StyleEncoding to create.

    Returns
    -------
    The StyleEncoding created from the given kwargs.

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


def _empty_array_like(single_array: StyleValue) -> StyleArray:
    shape = (0,) + single_array.shape
    return np.empty_like(single_array, shape=shape)


def _delete_in_bounds(array: np.ndarray, indices) -> np.ndarray:
    # We need to check bounds here because Points.remove_selected calls
    # delete once directly, then calls Points.data.setter which calls
    # delete again with OOB indices.
    safe_indices = [i for i in indices if i < array.shape[0]]
    return np.delete(array, safe_indices, axis=0)


def _broadcast_constant(
    constant: np.ndarray, n_rows: int, indices: Optional[IndicesType]
) -> np.ndarray:
    output_length = n_rows if indices is None else len(indices)
    output_shape = (output_length,) + constant.shape
    return np.broadcast_to(constant, output_shape)


def _maybe_index_array(
    array: np.ndarray, indices: Optional[IndicesType]
) -> np.ndarray:
    return array if indices is None else array[indices]
