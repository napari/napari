from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import numpy as np
import numpy.typing as npt

from napari.layers._data_protocols import LayerDataProtocol, assert_protocol
from napari.utils.translations import trans


# note: this also implements `LayerDataProtocol`, but we don't need to inherit.
class MultiScaleData(Sequence[LayerDataProtocol]):
    """Wrapper for multiscale data, to provide consistent API.

    :class:`LayerDataProtocol` is the subset of the python Array API that we
    expect array-likes to provide. Multiscale data is just a sequence of these
    array-likes.

    Parameters
    ----------
    data : Sequence[LayerDataProtocol]
        Levels of multiscale data, from larger to smaller.

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
    ) -> None:
        self._data: list[LayerDataProtocol] = list(data)
        if not self._data:
            raise ValueError(
                trans._('Multiscale data must be a (non-empty) sequence')
            )
        for d in self._data:
            assert_protocol(d, protocol=LayerDataProtocol)

    @property
    def size(self) -> int:
        """Size of the first scale."""
        return self._data[0].size

    @property
    def ndim(self) -> int:
        """ndim of the first scale."""
        return self._data[0].ndim

    @property
    def nlevels(self) -> int:
        """Number of multiscale levels."""
        return len(self._data)

    @property
    def dtype(self) -> npt.DTypeLike:
        """dtype of the first scale.."""
        return self._data[0].dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the first scale."""
        return self._data[0].shape

    @property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        """Tuple of shapes for all scales."""
        return tuple(im.shape for im in self._data)

    @overload
    def __getitem__(self, i: int) -> LayerDataProtocol: ...
    @overload
    def __getitem__(self, i: slice) -> Sequence[LayerDataProtocol]: ...
    def __getitem__(
        self, key: int | slice
    ) -> LayerDataProtocol | Sequence[LayerDataProtocol]:
        """Get individual multiscale levels."""
        return self._data[key]

    def __len__(self) -> int:
        """Number of multiscale levels."""
        return self.nlevels

    def __repr__(self) -> str:
        return (
            f'<MultiScaleData at {hex(id(self))}. '
            f"{len(self)} levels, '{self.dtype}', shapes: {self.shapes}>"
        )
