from __future__ import annotations

from types import GeneratorType
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ._data_protocols import LayerDataProtocol, assert_protocol
from .utils.layer_utils import compute_multiscale_level_and_corners


class MultiScaleData(Sequence[LayerDataProtocol], LayerDataProtocol):
    """Wrapper for multiscale data, to provide consistent API.

    :class:`LayerDataProtocol` is the subset of the python Array API that we
    expect array-likes to provide.  Multiscale data is just a sequence of
    array-likes (providing, e.g. `shape`, `dtype`, `__getitem__`).

    Parameters
    ----------
    data : Sequence[LayerDataProtocol]
        Levels of multiscale data, from larger to smaller.
    max_size : Sequence[int], optional
        Maximum size of a displayed tile in pixels, by default`data[-1].shape`

    Raises
    ------
    ValueError
        If `data` is empty or is not a list, tuple, or ndarray.
    TypeError
        If any of the items in `data` don't provide `LayerDataProtocol`.
    """

    def __init__(
        self,
        data: Sequence[LayerDataProtocol],
        max_size: Optional[Sequence[int]] = None,
    ) -> None:
        if isinstance(data, GeneratorType):
            data = list(data)
        if not (isinstance(data, (list, tuple, np.ndarray)) and len(data)):
            raise ValueError(
                "Multiscale data must be a (non-empty) list, tuple, or array"
            )
        for d in data:
            assert_protocol(d)

        self._data: Sequence[LayerDataProtocol] = data
        self.max_size = self._data[-1].shape if max_size is None else max_size
        self.downsample_factors = (
            np.array([d.shape for d in data]) / data[0].shape
        )

    @property
    def dtype(self) -> np.dtype:
        """Return dtype of the first scale.."""
        return self._data[0].dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of multiscale is just the biggest shape."""
        return self._data[0].shape

    @property
    def shapes(self) -> Tuple[Tuple[int, ...], ...]:
        """Tuple shapes for all scales."""
        return tuple(im.shape for im in self._data)

    def __getitem__(  # type: ignore [override]
        self, key: Union[int, Tuple[slice, ...]]
    ) -> LayerDataProtocol:
        """Multiscale indexing.

        When indexing with a single integer, the behavior is exactly like the
        input squence (it retrieves the sepcified level).

        When indexing with a tuple of slices, it will extract a chunk from the
        appropriate level using `compute_multiscale_level_and_corners`.

        All other `key` types are currently undefined and raise a
        NotImplementedError (for mixed tuple of slice and int), or TypeError
        (for everything else.)
        """
        if isinstance(key, int):
            return self._data[key]

        if isinstance(key, tuple):
            if all(isinstance(idx, slice) for idx in key):
                corners = np.array([(sl.start, sl.stop) for sl in key])
                level, corners = compute_multiscale_level_and_corners(
                    corners, self.max_size, self.downsample_factors
                )
                return self._data[level][corners]
            else:
                raise NotImplementedError("cannot handle slices and ints")

        raise TypeError(f"Cannot index multiscale data with {type(key)}")

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> bool:
        return self._data == other

    def __add__(self, other) -> bool:
        return self._data + other

    def __mul__(self, other) -> bool:
        return self._data * other

    def __rmul__(self, other) -> bool:
        return other * self._data

    def __array__(self) -> np.ndarray:
        return np.ndarray(self._data[-1])

    def __repr__(self) -> str:
        return (
            f"<MultiScaleData at {hex(id(self))}. "
            f"{len(self)} levels, '{self.dtype}', shapes: {self.shapes}>"
        )
