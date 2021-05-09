from __future__ import annotations

from types import GeneratorType
from typing import Any, List, Sequence, Tuple, TypeVar, Union

import numpy
from typing_extensions import Protocol, runtime_checkable

OBJ_NAMES = set(dir(Protocol))
OBJ_NAMES.update({'__annotations__', '__dict__', '__weakref__'})


def _raise_protocol_error(obj: Any, protocol: type):
    """Raise a more helpful error when required protocol members are missing."""
    needed = set(dir(protocol)).union(protocol.__annotations__) - OBJ_NAMES
    missing = needed - set(dir(obj))
    message = (
        f"Object of type {type(obj).__name__!r} does not implement "
        f"{protocol.__name__!r} Protocol.\n"
        f"Missing methods: {missing!r}"
    )
    raise TypeError(message)


def assert_protocol(obj: Any, protocol: type):
    if not isinstance(obj, protocol):
        _raise_protocol_error(obj, protocol)


Shape = Tuple[int, ...]


@runtime_checkable
class LayerDataProtocol(Protocol):
    shape: Shape
    dtype: numpy.dtype


@runtime_checkable
class MultiScaleDataProtocol(LayerDataProtocol, Protocol):
    def __getitem__(self, item) -> LayerDataProtocol:
        ...


_T = TypeVar('_T')
ListOrTuple = Union[List[_T], Tuple[_T, ...]]


class MultiScaleData(Sequence[LayerDataProtocol], MultiScaleDataProtocol):
    def __init__(self, data) -> None:
        if isinstance(data, GeneratorType):
            data = list(data)
        if not (isinstance(data, (list, tuple)) and data):
            raise ValueError(
                "Multiscale data must be a (non-empty) list or tuple."
            )
        for d in data:
            assert_protocol(d, LayerDataProtocol)

        self._data: ListOrTuple[LayerDataProtocol] = data

    @property
    def dtype(self):
        return self._data[0].dtype

    @property
    def shape(self):
        return self._data[0].shape

    @property
    def shapes(self) -> Tuple[Shape, ...]:
        return tuple(im.shape for im in self._data)

    def __getitem__(self, index):
        # TODO: handle slice indices
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
