import warnings
from abc import ABC, abstractmethod
from typing import Any, Generic, List, Optional, TypeVar, Union

import numpy as np
from typing_extensions import Protocol, runtime_checkable

from ...utils.events import EventedModel
from ...utils.translations import trans

IndicesType = Union[range, List[int], np.ndarray]

"""The variable type of a single style value."""
StyleValue = TypeVar('StyleValue', bound=np.ndarray)

"""The variable type of multiple style values in an array."""
StyleArray = TypeVar('StyleArray', bound=np.ndarray)


@runtime_checkable
class StyleEncoding(Protocol[StyleValue, StyleArray]):
    """Encodes style values, like colors and strings, from layer features.

    The public API of any StyleEncoding is just __call__, such that it can
    be called to generate style values from layer features. That call should
    be stateless, in that the values returned only depend on the given features.

    A StyleEncoding also has a private API that provides access to and mutation
    of previously generated and cached style values. This currently needs to be
    implemented to maintain some related behaviors in napari, but may be removed
    from this protocol in the future.
    """

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

    def _values(self) -> Union[StyleValue, StyleArray]:
        """Returns the previously generated and cached values.

        Returns
        -------
        Union[StyleValue, StyleArray]
            The cached values.
        """

    def _update(
        self, features: Any, *, indices: Optional[IndicesType] = None
    ) -> Union[StyleValue, StyleArray]:
        """Updates cached values by applying this to the tail of the given features.

        If the cached values have the same length as the given features, this may
        return the existing cached value array.

        Parameters
        ----------
        features : Dataframe-like
            The full layer features table from which to derive the output values.
        indices : Optional[IndicesType]
            The row indices for which to return values. If None, return all of them.

        Returns
        -------
        Union[StyleValue, StyleArray]
            The updated cached values, indexed by the indices if given.
        """

    def _append(self, array: StyleArray) -> None:
        """Appends raw style values to cached values.

        This is useful for supporting the paste operation in layers.

        Parameters
        ----------
        array : StyleArray
            The values to append. The dimensionality of these should match that of the existing style values.
        """

    def _delete(self, indices: IndicesType) -> None:
        """Deletes cached style values by index.

        Parameters
        ----------
        indices
            The indices of the style values to remove.
        """

    def _clear(self) -> None:
        """Clears all previously generated and cached values.

        Call this before calling _update this to refresh all cached values.
        """


class _ConstantStyleEncoding(EventedModel, Generic[StyleValue, StyleArray]):
    """Encodes a constant style value.

    Attributes
    ----------
    constant : StyleValue
        The constant style value.
    """

    constant: StyleValue

    def __call__(self, features: Any) -> Union[StyleValue, StyleArray]:
        return self.constant

    def _values(self) -> Union[StyleValue, StyleArray]:
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


class _ManualStyleEncoding(EventedModel, Generic[StyleValue, StyleArray]):
    """Encodes style values manually.

    The style values are encoded manually in the array attribute, so that
    attribute can be written to make persistent updates.

    Attributes
    ----------
    array : np.ndarray
        The array of values.
    default : np.ndarray
        The default style value that is used when ``array`` is shorter than
        the given features.
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

    def _values(self) -> Union[StyleValue, StyleArray]:
        return self.array

    def _update(
        self, features: Any, *, indices: Optional[IndicesType] = None
    ) -> Union[StyleValue, StyleArray]:
        self.array = self(features)
        return _maybe_index_array(self.array, indices)

    def _append(self, array: StyleArray) -> None:
        self.array = np.append(self.array, array, axis=0)

    def _delete(self, indices: IndicesType) -> None:
        self.array = np.delete(self.array, indices, axis=0)

    def _clear(self) -> None:
        pass


class _DerivedStyleEncoding(
    EventedModel, Generic[StyleValue, StyleArray], ABC
):
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

    @abstractmethod
    def __call__(self, features: Any) -> Union[StyleValue, StyleArray]:
        pass

    def _values(self) -> Union[StyleValue, StyleArray]:
        return self._cached

    def _update(
        self, features: Any, *, indices: Optional[IndicesType] = None
    ) -> Union[StyleValue, StyleArray]:
        n_cached = self._cached.shape[0]
        n_rows = features.shape[0]
        if n_cached < n_rows:
            tail_array = self._apply_safely(features.iloc[n_cached:n_rows])
            self._append(tail_array)
        elif n_cached > n_rows:
            self._cached = self._cached[:n_rows]
        return _maybe_index_array(self._cached, indices)

    def _apply_safely(self, features: Any) -> StyleArray:
        """Applies this without raising encoding errors, warning instead."""
        try:
            array = self(features.iloc)
        except (KeyError, ValueError):
            warnings.warn(
                trans._(
                    'Applying the encoding failed. Using the safe fallback value instead.',
                    deferred=True,
                ),
                category=RuntimeWarning,
            )
            shape = (features.shape[0],) + self.fallback.shape
            array = np.broadcast_to(self.fallback, shape)
        return array

    def _append(self, array: StyleArray) -> None:
        self._cached = np.append(self._cached, array, axis=0)

    def _delete(self, indices: IndicesType) -> None:
        self._cached = np.delete(self._cached, indices, axis=0)

    def _clear(self) -> None:
        self._cached = _empty_array_like(self.fallback)


def _empty_array_like(value: StyleValue) -> StyleArray:
    """Returns an empty array with the same type and remaining shape of the given value."""
    shape = (0,) + value.shape
    return np.empty_like(value, shape=shape)


def _maybe_index_array(
    array: StyleArray, indices: Optional[IndicesType]
) -> StyleArray:
    """Returns the given array, only using indices if given."""
    return array if indices is None else array[indices]
