"""Private type aliases shared by napari's annotations.

These are implementation details, not public API, and may change.
"""

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np

__all__ = ('Matrix2D', 'Vector1D')

# A 1D vector of numbers, e.g. a per-axis ``scale`` or ``translate``.
Vector1D: TypeAlias = (
    Sequence[float] | np.ndarray[tuple[int], np.dtype[np.number]]
)

# A 2D matrix of numbers, e.g. a ``rotate`` or ``shear`` matrix.
Matrix2D: TypeAlias = (
    Sequence[Sequence[float]]
    | np.ndarray[tuple[int, int], np.dtype[np.number]]
)
