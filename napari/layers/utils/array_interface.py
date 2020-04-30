import sys
from typing import Tuple, Union, Type

import numpy as np

if sys.version_info.minor < 8:
    from typing_extensions import Protocol
else:
    from typing import Protocol


class SupportsShape(Protocol):
    shape: Tuple[int, ...]


class Is2D(Protocol):
    shape: Tuple[int, int]


class SupportsLen(Protocol):
    def __len__(self, item) -> int:
        ...


class HasDtype(Protocol):
    dtype: Type[np.uint8]


class HasNdim(Protocol):
    @property
    def ndim(self) -> int:
        return 0


class BaseArrayInterface(SupportsLen, HasDtype, HasNdim, Protocol):
    def __getitem__(self, item) -> Union['ArrayInterface', int, float, bool]:
        ...


class ArrayInterface(BaseArrayInterface, SupportsShape, Protocol):
    ...


class Array2DInterface(BaseArrayInterface, Is2D):
    ...
