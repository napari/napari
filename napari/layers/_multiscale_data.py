from __future__ import annotations

import numbers
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ._data_protocols import LayerDataProtocol, assert_protocol

# from .utils.layer_utils import compute_multiscale_level_and_corners


HANDLED_FUNCTIONS = {}


def normalize_index(idx, ndim):
    if isinstance(idx, numbers.Integral):
        return (idx,) + (slice(None),) * (ndim - 1)
    elif isinstance(idx, tuple):
        new_idx = []
        for elem in idx:
            if elem is Ellipsis:
                elems = (
                    slice(
                        None,
                    )
                ) * (ndim - len(idx) + 1)
                new_idx.extend(elems)
            else:
                new_idx.append(elem)
        if len(new_idx) != ndim:
            new_idx.extend((slice(None),) * (ndim - len(idx)))
    return tuple(new_idx)


def make_array_indices_multiscale(index, downsample_factors):
    """Rescale the coordinates on each axis by the dowsample factors.

    This may result in some repeated indices in the lower-resolution levels.
    However, making them unique is probably more expensive and certainly much
    more complex than just ignoring the redundancy.
    """
    normalized_index = np.broadcast_arrays(*index)
    shape = normalized_index[0].shape
    ndim = len(shape)
    concatenated = np.stack(normalized_index, axis=0)
    rescaled = (
        concatenated * downsample_factors[(Ellipsis,) + (np.newaxis,) * ndim]
    )
    rescaled_int = rescaled.astype(int)
    return list(map(tuple, rescaled_int))


def scale_start_stop(value, factor):
    """Scale the start or stop of a slice by factor.

    This function accounts for the fact that start and stop can be None.
    """
    if value is None:
        return None
    return round(value * factor)


def scale_step(step, factor):
    """Scale the step in a slice by a given factor.

    This function accounts for the fact that step can be None, and that it
    must not be 0.
    """
    if step is None:
        return None
    return min(round(step * factor), 1)


def make_int_slice_indices_multiscale(key, downsample_factors):
    keys_by_dim = []
    for dimkey, factors in zip(key, downsample_factors.T):
        if isinstance(dimkey, int):
            dimkey_multiscale = np.round(dimkey * factors).astype(int)
        else:  # dimkey is a slice
            dimkey_multiscale = [
                slice(
                    scale_start_stop(dimkey.start, factor),
                    scale_start_stop(dimkey.stop, factor),
                    scale_step(dimkey.step, factor),
                )
                for factor in factors
            ]
        keys_by_dim.append(dimkey_multiscale)
    return list(zip(*keys_by_dim))


def make_index_multiscale(index, *, ndim, downsample_factors):
    key = normalize_index(index, ndim)
    types = set(map(type, key))
    if np.ndarray in types and slice in types:
        raise NotImplementedError(
            'Mixed array and slice indexing is not supported.'
        )
    elif np.ndarray in types:
        indices_by_level = make_array_indices_multiscale(
            key, downsample_factors
        )
    elif slice in types:
        indices_by_level = make_int_slice_indices_multiscale(
            key, downsample_factors
        )
    else:
        raise TypeError(
            f'Cannot index multiscale data with key {key} of type {type(key)}'
        )
    return indices_by_level


class MultiScaleData(LayerDataProtocol):
    """Wrapper for multiscale data, to provide Array API.

    :class:`LayerDataProtocol` is the subset of the python Array API that we
    expect array-likes to provide.  Multiscale data is just a sequence of
    array-likes (providing, e.g. `shape`, `dtype`, `__getitem__`).

    Parameters
    ----------
    data : Sequence[LayerDataProtocol]
        Levels of multiscale data, from larger to smaller.
    max_size : int, optional
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
        data: Union[Sequence[LayerDataProtocol], MultiScaleData],
        max_size: Optional[Sequence[int]] = None,
    ) -> None:
        self._data: List[LayerDataProtocol] = list(data)
        if not self._data:
            raise ValueError("Multiscale data must be a (non-empty) sequence")
        for d in self._data:
            assert_protocol(d)

        self._max_size = 0
        self.compute_level = -1
        if max_size is not None:
            self.max_size = max_size
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
    def ndim(self) -> int:
        return self._data[0].ndim

    @property
    def shapes(self) -> Tuple[Tuple[int, ...], ...]:
        """Tuple shapes for all scales."""
        return tuple(im.shape for im in self._data)

    @property
    def max_size(self) -> int:
        return self._max_size

    @max_size.setter
    def max_size(self, maxsize):
        """Don't use more that `maxsize` elements when computing min/max/etc.

        This convenience function calls set_compute_at_level, but uses the
        size (in number of array elements) of the computation an the deciding
        factor, rather than a specific level in the hierarchy (whose size may
        not be known ahead of time).
        """
        self._max_size = maxsize
        data_sizes = np.array([arr.size for arr in self._data])
        usable = data_sizes < maxsize
        self._compute_at_level = np.argmax(data_sizes * usable)

    @property
    def compute_level(self):
        return self._compute_at_level

    @compute_level.setter
    def compute_level(self, lvl):
        """Compute functions such as np.min and np.max at this level.

        We support a subset of NumPy functions to help in certain operations,
        such as computing contrast limits. Since exact min/max values may not
        be preserved at different levels of an image pyramid, we create a
        stateful switch to determine on which array in ``_data`` the NumPy
        function is run.
        """
        self.max_size = self._data[lvl].size + 1

    @property
    def current_level_data(self):
        return self._data[self.compute_level]

    def __getitem__(  # type: ignore [override]
        self, index: Union[int, Tuple[slice, ...]]
    ) -> MultiScaleData:
        """Multiscale indexing.

        This is intended to behave like normal array indexing of the highest-
        resolution scale, but it returns a new multiscale array for those
        indices.

        We support the following types of indexing:
        - integer indexing: we return only a single value as expected.
        - mixed slices and integers: this behaves like NumPy indexing: the
          dimensions corresponding to integers (if any) are dropped, the
          sliced ones create a new multiscale array around those slices.
        - array indexing: this behaves like NumPy fancy indexing, but returns
          translates the indices to multiple scales and returns the values at
          each scale.
        - mixed array and slice indexing is NOT SUPPORTED.
        """
        indices_by_level = make_index_multiscale(
            index, ndim=self.ndim, downsample_factors=self.downsample_factors
        )
        new_data = [dat[idx] for dat, idx in zip(self._data, indices_by_level)]
        return MultiScaleData(new_data)

    def __setitem__(  # type: ignore [override]
        self, index: Union[int, Tuple[slice, ...]], value
    ) -> MultiScaleData:
        """Multiscale indexing.

        This is intended to behave like normal array indexing of the highest-
        resolution scale, but it returns a new multiscale array for those
        indices.

        We support the following types of indexing:
        - integer indexing: we return only a single value as expected.
        - mixed slices and integers: this behaves like NumPy indexing: the
          dimensions corresponding to integers (if any) are dropped, the
          sliced ones create a new multiscale array around those slices.
        - array indexing: this behaves like NumPy fancy indexing, but returns
          translates the indices to multiple scales and returns the values at
          each scale.
        - mixed array and slice indexing is NOT SUPPORTED.
        """
        indices_by_level = make_index_multiscale(
            index, ndim=self.ndim, downsample_factors=self.downsample_factors
        )
        for dat, idx in zip(self._data, indices_by_level):
            dat[idx] = value

    def __len__(self) -> int:
        return len(self._data)

    def __array__(self) -> np.ndarray:
        return np.ndarray(self._data[self._compute_at_level])

    def __repr__(self) -> str:
        return (
            f"<MultiScaleData at {hex(id(self))}. "
            f"{len(self)} levels, '{self.dtype}', shapes: {self.shapes}>"
        )

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            new_args = []
            for arg in args:
                if isinstance(arg, MultiScaleData):
                    new_args.append(arg.current_level_data)
                else:
                    new_args.append(arg)
            return func(*new_args, **kwargs)
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MultiScaleData objects
        if not all(issubclass(t, MultiScaleData) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def level(self, lvl: int):
        """Return the array at level `lvl`."""
        return self._data[lvl]

    def set_compute_at_level(self, lvl: int):
        level_size = self._data[lvl].size
        self.max_size = level_size + 1
