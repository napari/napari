import sys
from typing import Tuple, Union

import numpy as np

if sys.version_info.minor < 8:
    from typing_extensions import Protocol
else:
    from typing import Protocol


class ArrayInterface(Protocol):
    shape: Tuple[int] = (0,)
    dtype: type = np.uint8
    ndim: int = 1

    def __getitem__(self, item) -> Union['ArrayInterface', int, float, bool]:
        ...

    def __len__(self) -> int:
        ...
