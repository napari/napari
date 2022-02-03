import warnings
from abc import ABC, abstractmethod
from typing import Any, Generic, List, Optional, TypeVar, Union

import numpy as np

from ...utils.events import EventedModel
from ...utils.translations import trans

IndicesType = Union[range, List[int], np.ndarray]

"""The variable type of a single style value."""
StyleValue = TypeVar('StyleValue', bound=np.ndarray)

"""The variable type of multiple style values in an array."""
StyleArray = TypeVar('StyleArray', bound=np.ndarray)


class StyleEncoding(EventedModel, Generic[StyleValue, StyleArray], ABC):
    """Defines a way to encode style values, like colors and strings."""

    @abstractmethod
    def __call__(self, features: Any) -> Union[StyleValue, StyleArray]:
        """Apply this encoding with the given features to generate style values.

        Parameters
        ----------
        features : Dataframe-like
            The layer features table from which to derive the output values.

        Returns
        -------
        Union[StyleValue, StyleArray]
            Either a single style value (e.g. from a constant encoding) or an
            array of encoded values the same length as the given features.

        Raises
        ------
        KeyError, ValueError
            If generating values from the given features fails.
        """

    @abstractmethod
    def _update(
        self, features: Any, *, indices: Optional[IndicesType] = None
    ) -> Union[StyleValue, StyleArray]:
        """Updates cached values by applying this to the tail of the given features.

        If the cached values have the same length as the given features, this may
        return the existing cached value array.

        If generating values from the given features fails, this will fall back
        to returning some safe or default value.

        Parameters
        ----------
        features : Dataframe-like
            The layer features table from which to derive the output values.
        indices : Optional[IndicesType]
            The row indices for which to return values. If None, return all of them.

        Returns
        -------
            The updated array of cached values, possibly indexed by the given indices.
        """

    @abstractmethod
    def _append(self, array: StyleArray) -> None:
        """Appends raw style values to cached values.

        This is useful for supporting the paste operation in layers.

        Parameters
        ----------
        array : StyleArray
            The values to append. The dimensionality of these should match that of the existing style values.
        """

    @abstractmethod
    def _delete(self, indices: IndicesType) -> None:
        """Deletes cached style values by index.

        Parameters
        ----------
        indices
            The indices of the style values to remove.
        """

    @abstractmethod
    def _clear(self) -> None:
        """Clears all previously generated and cached values.

        Call this before calling _update this to refresh all cached values.
        """


class ConstantStyleEncoding(StyleEncoding[StyleValue, StyleArray]):
    """Encodes a constant style value.

    Attributes
    ----------
    constant : StyleValue
        The constant style value.
    """

    constant: StyleValue

    def __call__(self, features: Any) -> Union[StyleValue, StyleArray]:
        return self.constant

    def _update(
        self, features: Any, *, indices: Optional[IndicesType] = None
    ) -> Union[StyleValue, StyleArray]:
        return self.constant

    def _append(self, array: StyleArray) -> None:
        pass

    def _delete(self, indices: IndicesType) -> None:
        pass

    def _clear(self) -> None:
        pass


class ManualStyleEncoding(StyleEncoding[StyleValue, StyleArray]):
    """Encodes style values manually.

    The style values are encoded manually in the array attribute, so that
    attribute can be written to make persistent updates.

    Attributes
    ----------
    array : StyleArray
        The array of values.
    default : StyleValue
        The default style value that is used when requesting a value that
        is out of bounds in the array attribute.
    """

    array: StyleArray
    default: StyleValue

    def __call__(self, features: Any) -> Union[StyleArray, StyleValue]:
        n_values = self.array.shape[0]
        n_rows = features.shape[0]
        if n_rows > n_values:
            tail_array = np.array([self.default] * (n_rows - n_values))
            return np.append(self.array, tail_array, axis=0)
        return np.array(self.array[:n_rows])

    def _update(
        self, features: Any, *, indices: Optional[IndicesType] = None
    ) -> Union[StyleValue, StyleArray]:
        n_values = len(self.array)
        n_rows = features.shape[0]
        if n_values < n_rows:
            self.array = self(features)
        elif n_values > n_rows:
            self.array = self.array[:n_rows]
        return _maybe_index_array(self.array, indices)

    def _append(self, array: StyleArray) -> None:
        self.array = np.append(self.array, array, axis=0)

    def _delete(self, indices: IndicesType) -> None:
        self.array = _delete_in_bounds(self.array, indices)

    def _clear(self) -> None:
        pass


class DerivedStyleEncoding(StyleEncoding[StyleValue, StyleArray], ABC):
    """Encodes style values by deriving them from feature values.

    Attributes
    ----------
    fallback : StyleValue
        The fallback style value.
    """

    fallback: StyleValue
    _cached: StyleArray

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cached = _empty_array_like(self.fallback)

    def _update(
        self, features: Any, *, indices: Optional[IndicesType] = None
    ) -> Union[StyleValue, StyleArray]:
        n_cached = self._cached.shape[0]
        n_rows = features.shape[0]
        if n_cached < n_rows:
            try:
                tail_array = self(features.iloc[n_cached:n_rows])
            except (KeyError, ValueError):
                warnings.warn(
                    trans._(
                        'Applying the encoding failed. Using a safe fallback value instead.',
                        deferred=True,
                    ),
                    category=RuntimeWarning,
                )
                tail_shape = (n_rows - n_cached,) + self.fallback.shape
                tail_array = np.broadcast_to(self.fallback, tail_shape)
            self._append(tail_array)
        elif n_cached > n_rows:
            self._cached = self._cached[:n_rows]
        return _maybe_index_array(self._cached, indices)

    def _append(self, array: StyleArray) -> None:
        self._cached = np.append(self._cached, array, axis=0)

    def _delete(self, indices: IndicesType) -> None:
        self._cached = _delete_in_bounds(self._cached, indices)

    def _clear(self) -> None:
        self._cached = _empty_array_like(self.fallback)


def _empty_array_like(single_array: StyleValue) -> StyleArray:
    shape = (0,) + single_array.shape
    return np.empty_like(single_array, shape=shape)


def _delete_in_bounds(array: np.ndarray, indices: IndicesType) -> np.ndarray:
    # We need to check bounds here because Points.remove_selected calls
    # delete once directly, then calls Points.data.setter which calls
    # delete again with OOB indices.
    if isinstance(indices, range):
        safe_indices = range(indices.start, array.shape[0], indices.step)
    else:
        safe_indices = [i for i in indices if i < array.shape[0]]
    return np.delete(array, safe_indices, axis=0)


def _maybe_index_array(
    array: np.ndarray, indices: Optional[IndicesType]
) -> np.ndarray:
    return array if indices is None else array[indices]
