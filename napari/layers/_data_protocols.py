"""This module holds Protocols that layer.data objects are expected to provide.
"""
from __future__ import annotations

from types import GeneratorType
from typing import Any, List, Sequence, Tuple, TypeVar, Union

import numpy as np
from typing_extensions import Protocol, runtime_checkable

_T = TypeVar('_T')
Shape = Tuple[int, ...]
ListOrTuple = Union[List[_T], Tuple[_T, ...], np.ndarray]
_OBJ_NAMES = set(dir(Protocol))
_OBJ_NAMES.update({'__annotations__', '__dict__', '__weakref__'})


def _raise_protocol_error(obj: Any, protocol: type):
    """Raise a more helpful error when required protocol members are missing."""
    needed = set(dir(protocol)).union(protocol.__annotations__) - _OBJ_NAMES
    missing = needed - set(dir(obj))
    message = (
        f"Object of type {type(obj).__name__!r} does not implement "
        f"{protocol.__name__!r} Protocol.\n"
        f"Missing methods: {missing!r}"
    )
    raise TypeError(message)


def assert_protocol(obj: Any, protocol: type):
    """Assert `obj` is an instance of `protocol` or raise helpful error."""
    if not isinstance(obj, protocol):
        _raise_protocol_error(obj, protocol)


@runtime_checkable
class LayerDataProtocol(Protocol):
    """A Protocol that all layer.data needs to support.

    WIP: Shapes.data may be an execption.
    """

    shape: Shape
    dtype: np.dtype

    def __getitem__(self, item) -> LayerDataProtocol:
        ...


class MultiScaleData(Sequence[LayerDataProtocol], LayerDataProtocol):
    """Wrapper for multiscale data, to provide consistent API."""

    def __init__(self, data) -> None:
        if isinstance(data, GeneratorType):
            data = list(data)
        if not (isinstance(data, (list, tuple, np.ndarray)) and len(data)):
            raise ValueError(
                "Multiscale data must be a (non-empty) list, tuple, or array"
            )
        for d in data:
            assert_protocol(d, LayerDataProtocol)

        self._data: ListOrTuple[LayerDataProtocol] = data

    @property
    def dtype(self):
        """Return dtype of the first scale.."""
        return self._data[0].dtype

    @property
    def shape(self):
        """Shape of multiscale is just the biggest shape."""
        return self._data[0].shape

    @property
    def shapes(self) -> Tuple[Shape, ...]:
        """Tuple shapes for all scales."""
        return tuple(im.shape for im in self._data)

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> bool:
        return self._data == other

    def _add__(self, other) -> bool:
        return self._data + other

    def __mul__(self, other) -> bool:
        return self._data * other

    def __rmul__(self, other) -> bool:
        return other * self._data
