import sys
from typing import Tuple, Union

import numpy as np

if sys.version_info.minor < 8:
    from typing_extensions import Protocol
else:
    from typing import Protocol


class ArrayInterface(Protocol):
    def __getitem__(self, item) -> Union['ArrayInterface', int, float, bool]:
        ...

    def __len__(self) -> int:
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        return (0,)

    @property
    def dtype(self) -> type:
        return np.uint8

    @property
    def ndim(self) -> int:
        return 0
