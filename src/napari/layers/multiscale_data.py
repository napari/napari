from __future__ import annotations

import itertools
import warnings
from collections.abc import Sequence
from typing import overload

import numpy as np
import numpy.typing as npt

from napari.layers._data_protocols import LayerDataProtocol, assert_protocol


def validate_multiscale_data(
    data: Sequence[LayerDataProtocol],
) -> list[LayerDataProtocol]:
    """Validate that `data` is a well-formed sequence of multiscale levels.

    Checks that `data` is non-empty, that every level implements
    :class:`LayerDataProtocol`, that all levels have the same number of
    dimensions, and that level sizes are non-increasing (equal-size levels
    are allowed, e.g. as produced by projecting a multiscale image along
    an axis where different levels aren't downsampled).

    Parameters
    ----------
    data : Sequence[LayerDataProtocol]
        Levels of multiscale data, from larger to smaller.

    Returns
    -------
    list[LayerDataProtocol]
        `data` coerced to a list.

    Raises
    ------
    ValueError
        If `data` is empty.
    TypeError
        If any item in `data` does not implement `LayerDataProtocol`
        (raised by :func:`assert_protocol`).
    ValueError
        If the items in `data` do not all have the same `ndim`.
    ValueError
        If the `size` of the items in `data` is not non-increasing.
    """
    data = list(data)
    if not data:
        raise ValueError('Multiscale data must be a (non-empty) sequence')
    sizes = []
    ndims = []
    for d in data:
        assert_protocol(d, protocol=LayerDataProtocol)
        sizes.append(d.size)
        ndims.append(d.ndim)

    if any(n != ndims[0] for n in ndims):
        raise ValueError(
            f'Input data should be a sequence of array-like objects with '
            f'the same number of dimensions. Got ndims: {ndims}'
        )

    non_increasing = all(s1 >= s2 for s1, s2 in itertools.pairwise(sizes))
    if not non_increasing:
        raise ValueError(
            f'Input data should be a sequence of array-like objects of non-increasing size. Got arrays in incorrect order, sizes: {sizes}'
        )
    return data


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

        self._data: list[LayerDataProtocol] = validate_multiscale_data(data)

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

    def get_level(self, i: int) -> LayerDataProtocol:
        """Get the array-like data at resolution level `i`.

        Parameters
        ----------
        i : int
            Resolution level of data to return.

        Returns
        -------
        LayerDataProtocol
            The array-like data at resolution level `i`.

        Raises
        ------
        ValueError
            If `i` is out of bounds of the resolution levels.
        """
        if i >= self.nlevels:
            raise ValueError(
                f'Level {i} out of bounds for {self.nlevels} multiscale levels'
            )
        return self._data[i]

    @overload
    def __getitem__(self, i: int) -> LayerDataProtocol: ...
    @overload
    def __getitem__(self, i: slice) -> Sequence[LayerDataProtocol]: ...
    def __getitem__(
        self, key: int | slice
    ) -> LayerDataProtocol | Sequence[LayerDataProtocol]:
        """Get individual multiscale levels."""
        return self._data[key]

    def __array__(self) -> npt.NDArray:
        """Get numpy array of the lowest resolution level."""
        warnings.warn(
            'MultiScaleData.__array__ gives you the lowest resolution, while MultiScaleData.shape gives you the high resolution shape. Use MultiScaleData.get_level() to get a specific resolution level'
        )
        return np.asarray(self._data[-1])

    def __len__(self) -> int:
        """Number of multiscale levels."""
        return self.nlevels

    def __eq__(self, other: object) -> bool:
        return self._data == other

    def __repr__(self) -> str:
        return (
            f'<MultiScaleData at {hex(id(self))}. '
            f"{len(self)} levels, '{self.dtype}', shapes: {self.shapes}>"
        )
