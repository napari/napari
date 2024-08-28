import logging
import warnings
from abc import ABC, abstractmethod
from typing import (
    Any,
    Generic,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np

from napari.utils.events import EventedModel
from napari.utils.translations import trans

IndicesType = Union[range, list[int], np.ndarray]

"""The variable type of a single style value."""
StyleValue = TypeVar('StyleValue', bound=np.ndarray)

"""The variable type of multiple style values in an array."""
StyleArray = TypeVar('StyleArray', bound=np.ndarray)


@runtime_checkable
class StyleEncoding(Protocol[StyleValue, StyleArray]):
    """Encodes generic style values, like colors and strings, from layer features.

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

    @property
    def _values(self) -> Union[StyleValue, StyleArray]:
        """The previously generated and cached values."""

    def _apply(self, features: Any) -> None:
        """Applies this to the tail of the given features and updates cached values.

        If the cached values are longer than the given features, this will remove
        the extra cached values. If they are the same length, this may do nothing.

        Parameters
        ----------
        features : Dataframe-like
            The full layer features table from which to derive the output values.
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
        """Clears all previously generated and cached values."""

    def _json_encode(self) -> dict:
        """Convert this to a dictionary that can be passed to json.dumps.

        Returns
        -------
        dict
            The dictionary representation of this with JSON compatible keys and values.
        """


class StyleCollection(EventedModel):
    """Base class for mapping channels to encodings.

    Extend this class with an attribute named after each visual channel
    containing the encoding for that channel.

    Examples
    --------
    >>> import pandas as pd
    >>> class Style(StyleCollection):
    >>>     edge_color: ColorEncoding
    >>>     face_color: ColorEncoding
    >>> style = Style(
    >>>     edge_color={
    >>>         'feature': 'confidence',
    >>>         'colormap': 'gray',
    >>>         'contrast_limits': (0, 1),
    >>>     },
    >>>     face_color={
    >>>         'feature': 'good_point',
    >>>         'colormap': {False: 'green', True: 'blue'},
    >>>     },
    >>> )
    >>> features = pd.DataFrame({
    >>>     'confidence': [1, 0.5, 0],
    >>>     'good_point': [True, False, False],
    >>> })
    >>> style(features)
    {
        'edge_color': array([[1, 1, 1, 1], [0.5, 0.5, 0.5, 1], [0, 0, 0, 1]]),
        'face_color': array([[0, 1, 0, 1], [0, 0, 1, 1]]),
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Implement event bubbling from style encoding instances by connecting
        # each encoding's main event to this collection's main event.
        for encoding in self._encodings:
            encoding.events.connect(self.events)

    def __setattr__(self, name: str, value: Any) -> None:
        # Maintain event bubbling from new instance of encoding.
        is_channel = name in self._channels
        logging.warning(
            'StyleCollection.__setattr__: %s, %s, %s', name, value, is_channel
        )
        if is_channel:
            getattr(self, name).events.disconnect(self.events)
        super().__setattr__(name, value)
        if is_channel:
            getattr(self, name).events.connect(self.events)

    def __call__(
        self, features: Any
    ) -> dict[str, Union[StyleValue, StyleArray]]:
        """Apply all encodings with the given features to generate style values.

        Parameters
        ----------
        features : Dataframe-like
            The layer features table from which to derive the output values.

        Returns
        -------
        dict[str, Union[StyleValue, StyleArray]]
            Maps from channel/field name to either a single style value
            (e.g. from a constant encoding) or an array of encoded values the
            same length as the given features.

        Raises
        ------
        KeyError, ValueError
            If generating values from the given features fails.
        """
        return {
            channel: encoding(features)
            for channel, encoding in zip(self._channels, self._encodings)
        }

    @property
    def _channels(self) -> tuple[str, ...]:
        """Channel names in this style collection.

        Example: ('face_color', 'edge_color', 'size').
        """
        return tuple(self.__fields__)

    @property
    def _encodings(self) -> tuple[StyleEncoding, ...]:
        """Encodings in this style collection, one per visual channel.

        The order matches the order of `_channels`.
        """
        return tuple(
            getattr(self, channel_name) for channel_name in self._channels
        )

    def _apply(self, features: Any) -> None:
        # TODO: what if instead of storing the cached values in the encodings,
        # we stored them in the collection. This would allow us to simplify
        # the style encoding and also potentially store features in the collection
        # to more easily respond to the necessary changes.
        # This may also allow an easy separation of encoding and value store,
        # which was desired but was difficult to implement prior to introducing
        # the collection.
        for encoding in self._encodings:
            encoding._apply(features)

    def _clear(self) -> None:
        for encoding in self._encodings:
            encoding._clear()

    def _refresh(self, features: Any) -> None:
        self._clear()
        self._apply(features)

    def _delete(self, indices: IndicesType) -> None:
        for encoding in self._encodings:
            encoding._delete(indices)

    def _copy(
        self, indices: IndicesType
    ) -> dict[str, Union[StyleValue, StyleArray]]:
        return {
            channel: _get_style_values(encoding, indices)
            for channel, encoding in zip(self._channels, self._encodings)
        }

    def _paste(self, **elements) -> None:
        for channel, values in elements.items():
            getattr(self, channel)._append(values)


class _StyleEncodingModel(EventedModel):
    class Config:
        # Forbid extra initialization parameters instead of ignoring
        # them by default. This is useful when parsing style encodings
        # from dicts, as different types of encodings may have the same
        # field names.
        # https://pydantic-docs.helpmanual.io/usage/model_config/#options
        extra = 'forbid'


# The following classes provide generic implementations of common ways
# to encode style values, like constant, manual, and derived encodings.
# They inherit Python's built-in `Generic` type, so that an encoding with
# a specific output type can inherit the generic type annotations from
# this class along with the functionality it provides. For example,
# `ConstantStringEncoding.__call__` returns an `Array[str, ()]` whereas
# `ConstantColorEncoding.__call__` returns an `Array[float, (4,)]`.
# For more information on `Generic`, see the official docs.
# https://docs.python.org/3/library/typing.html#generics


class _ConstantStyleEncoding(
    _StyleEncodingModel, Generic[StyleValue, StyleArray]
):
    """Encodes a constant style value.

    This encoding is generic so that it can be used to implement style
    encodings with different value types like Array[]

    Attributes
    ----------
    constant : StyleValue
        The constant style value.
    """

    constant: StyleValue

    def __call__(self, features: Any) -> Union[StyleValue, StyleArray]:
        return self.constant

    @property
    def _values(self) -> Union[StyleValue, StyleArray]:
        return self.constant

    def _apply(self, features: Any) -> None:
        pass

    def _append(self, array: StyleArray) -> None:
        pass

    def _delete(self, indices: IndicesType) -> None:
        pass

    def _clear(self) -> None:
        pass

    def _json_encode(self) -> dict:
        return self.dict()


class _ManualStyleEncoding(
    _StyleEncodingModel, Generic[StyleValue, StyleArray]
):
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

    @property
    def _values(self) -> Union[StyleValue, StyleArray]:
        return self.array

    def _apply(self, features: Any) -> None:
        self.array = self(features)

    def _append(self, array: StyleArray) -> None:
        self.array = np.append(self.array, array, axis=0)

    def _delete(self, indices: IndicesType) -> None:
        self.array = np.delete(self.array, indices, axis=0)

    def _clear(self) -> None:
        pass

    def _json_encode(self) -> dict:
        return self.dict()


class _DerivedStyleEncoding(
    _StyleEncodingModel, Generic[StyleValue, StyleArray], ABC
):
    """Encodes style values by deriving them from feature values.

    Attributes
    ----------
    fallback : StyleValue
        The fallback style value.
    """

    fallback: StyleValue
    _cached: StyleArray

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cached = _empty_array_like(self.fallback)

    @abstractmethod
    def __call__(self, features: Any) -> Union[StyleValue, StyleArray]:
        pass

    # This is a crude way to clear cached values when any of the fields
    # that affect those values change.
    def __setattr__(self, name: str, value: Any) -> None:
        if name[0] != '_' and name != 'fallback':
            self._clear()
        super().__setattr__(name, value)

    @property
    def _values(self) -> Union[StyleValue, StyleArray]:
        return self._cached

    def _apply(self, features: Any) -> None:
        n_cached = self._cached.shape[0]
        n_rows = features.shape[0]
        if n_cached < n_rows:
            tail_array = self._call_safely(features.iloc[n_cached:n_rows])
            self._append(tail_array)
        elif n_cached > n_rows:
            self._cached = self._cached[:n_rows]

    def _call_safely(self, features: Any) -> StyleArray:
        """Calls this without raising encoding errors, warning instead."""
        try:
            array = self(features)
        except (KeyError, ValueError):
            warnings.warn(
                trans._(
                    'Applying the encoding failed. Using the safe fallback value instead.',
                    deferred=True,
                ),
                category=RuntimeWarning,
            )
            shape = (features.shape[0], *self.fallback.shape)
            array = np.broadcast_to(self.fallback, shape)
        return array

    def _append(self, array: StyleArray) -> None:
        self._cached = np.append(self._cached, array, axis=0)

    def _delete(self, indices: IndicesType) -> None:
        self._cached = np.delete(self._cached, indices, axis=0)

    def _clear(self) -> None:
        self._cached = _empty_array_like(self.fallback)

    def _json_encode(self) -> dict:
        return self.dict()


def _get_style_values(
    encoding: StyleEncoding[StyleValue, StyleArray],
    indices: IndicesType,
    value_ndim: int = 0,
):
    """Returns a scalar style value or indexes non-scalar style values."""
    values = encoding._values
    return values if values.ndim == value_ndim else values[indices]


def _empty_array_like(value: StyleValue) -> StyleArray:
    """Returns an empty array with the same type and remaining shape of the given value."""
    shape = (0, *value.shape)
    return np.empty_like(value, shape=shape)
