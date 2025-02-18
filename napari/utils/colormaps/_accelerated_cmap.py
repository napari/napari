"""
Colormap utility functions to be sped-up by numba JIT.

These should stay in a separate module because they need to be reloaded during
testing, which can break instance/class relationships when done dynamically.
See https://github.com/napari/napari/pull/7025#issuecomment-2186190719.
"""

from importlib.metadata import version
from typing import TYPE_CHECKING

import numpy as np
from packaging.version import parse

if TYPE_CHECKING:
    from numba import typed

    from napari.utils.colormaps import DirectLabelColormap


__all__ = (
    'labels_raw_to_texture_direct',
    'minimum_dtype_for_labels',
    'zero_preserving_modulo',
    'zero_preserving_modulo_numpy',
)

MAPPING_OF_UNKNOWN_VALUE = 0
# For direct mode we map all unknown values to single value
# for simplicity of implementation we select 0


def minimum_dtype_for_labels(num_colors: int) -> np.dtype:
    """Return the minimum texture dtype that can hold given number of colors.

    Parameters
    ----------
    num_colors : int
        Number of unique colors in the data.

    Returns
    -------
    np.dtype
        Minimum dtype that can hold the number of colors.
    """
    if num_colors <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if num_colors <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.float32)


def zero_preserving_modulo_numpy(
    values: np.ndarray, n: int, dtype: np.dtype, to_zero: int = 0
) -> np.ndarray:
    """``(values - 1) % n + 1``, but with one specific value mapped to 0.

    This ensures (1) an output value in [0, n] (inclusive), and (2) that
    no nonzero values in the input are zero in the output, other than the
    ``to_zero`` value.

    Parameters
    ----------
    values : np.ndarray
        The dividend of the modulo operator.
    n : int
        The divisor.
    dtype : np.dtype
        The desired dtype for the output array.
    to_zero : int, optional
        A specific value to map to 0. (By default, 0 itself.)

    Returns
    -------
    np.ndarray
        The result: 0 for the ``to_zero`` value, ``values % n + 1``
        everywhere else.
    """
    if n > np.iinfo(dtype).max:
        # n is to big, modulo will be pointless
        res = values.astype(dtype)
        res[values == to_zero] = 0
        return res
    res = ((values - 1) % n + 1).astype(dtype)
    res[values == to_zero] = 0
    return res


def _zero_preserving_modulo_loop(
    values: np.ndarray, n: int, dtype: np.dtype, to_zero: int = 0
) -> np.ndarray:
    """``(values - 1) % n + 1``, but with one specific value mapped to 0.

    This ensures (1) an output value in [0, n] (inclusive), and (2) that
    no nonzero values in the input are zero in the output, other than the
    ``to_zero`` value.

    Parameters
    ----------
    values : np.ndarray
        The dividend of the modulo operator.
    n : int
        The divisor.
    dtype : np.dtype
        The desired dtype for the output array.
    to_zero : int, optional
        A specific value to map to 0. (By default, 0 itself.)

    Returns
    -------
    np.ndarray
        The result: 0 for the ``to_zero`` value, ``values % n + 1``
        everywhere else.
    """
    result = np.empty_like(values, dtype=dtype)
    # need to preallocate numpy array for asv memory benchmarks
    return _zero_preserving_modulo_inner_loop(values, n, to_zero, out=result)


def _zero_preserving_modulo_inner_loop(
    values: np.ndarray, n: int, to_zero: int, out: np.ndarray
) -> np.ndarray:
    """``(values - 1) % n + 1``, but with one specific value mapped to 0.

    This ensures (1) an output value in [0, n] (inclusive), and (2) that
    no nonzero values in the input are zero in the output, other than the
    ``to_zero`` value.

    Parameters
    ----------
    values : np.ndarray
        The dividend of the modulo operator.
    n : int
        The divisor.
    to_zero : int
        A specific value to map to 0. (Usually, 0 itself.)
    out : np.ndarray
        Preallocated output array

    Returns
    -------
    np.ndarray
        The result: 0 for the ``to_zero`` value, ``values % n + 1``
        everywhere else.
    """
    for i in prange(values.size):
        if values.flat[i] == to_zero:
            out.flat[i] = 0
        else:
            out.flat[i] = (values.flat[i] - 1) % n + 1

    return out


if parse('2.0') <= parse(version('numpy')) < parse('2.1'):

    def clip(data: np.ndarray, min_val: int, max_val: int) -> np.ndarray:
        """
        Clip data to the given range.
        """
        dtype_info = np.iinfo(data.dtype)
        min_val = max(min_val, dtype_info.min)
        max_val = min(max_val, dtype_info.max)
        return np.clip(data, min_val, max_val)
else:

    def clip(data: np.ndarray, min_val: int, max_val: int) -> np.ndarray:
        return np.clip(data, min_val, max_val)


def _labels_raw_to_texture_direct_numpy(
    data: np.ndarray, direct_colormap: 'DirectLabelColormap'
) -> np.ndarray:
    """Convert labels data to the data type used in the texture.

    This implementation uses numpy vectorized operations.

    See `_cast_labels_data_to_texture_dtype_direct` for more details.
    """
    if direct_colormap.use_selection:
        return (data == direct_colormap.selection).astype(np.uint8)
    mapper = direct_colormap._array_map
    if any(x < 0 for x in direct_colormap.color_dict if x is not None):
        half_shape = mapper.shape[0] // 2 - 1
        data = clip(data, -half_shape, half_shape)
    else:
        data = clip(data, 0, mapper.shape[0] - 1)

    return mapper[data]


def _labels_raw_to_texture_direct_loop(
    data: np.ndarray, direct_colormap: 'DirectLabelColormap'
) -> np.ndarray:
    """
    Cast direct labels to the minimum type.

    Parameters
    ----------
    data : np.ndarray
        The input data array.
    direct_colormap : DirectLabelColormap
        The direct colormap.

    Returns
    -------
    np.ndarray
        The cast data array.
    """
    if direct_colormap.use_selection:
        return (data == direct_colormap.selection).astype(np.uint8)

    dkt = direct_colormap._get_typed_dict_mapping(data.dtype)
    target_dtype = minimum_dtype_for_labels(
        direct_colormap._num_unique_colors + 2
    )
    result_array = np.full_like(
        data, MAPPING_OF_UNKNOWN_VALUE, dtype=target_dtype
    )
    return _labels_raw_to_texture_direct_inner_loop(data, dkt, result_array)


def _labels_raw_to_texture_direct_inner_loop(
    data: np.ndarray, dkt: 'typed.Dict', out: np.ndarray
) -> np.ndarray:
    """
    Relabel data using typed dict with mapping unknown labels to default value
    """
    # The numba typed dict does not provide official Api for
    # determine key and value types
    for i in prange(data.size):
        val = data.flat[i]
        if val in dkt:
            out.flat[i] = dkt[data.flat[i]]

    return out


try:
    import numba
except ModuleNotFoundError:
    zero_preserving_modulo = zero_preserving_modulo_numpy
    labels_raw_to_texture_direct = _labels_raw_to_texture_direct_numpy
    prange = range
else:
    _zero_preserving_modulo_inner_loop = numba.njit(parallel=True, cache=True)(
        _zero_preserving_modulo_inner_loop
    )
    zero_preserving_modulo = _zero_preserving_modulo_loop
    labels_raw_to_texture_direct = _labels_raw_to_texture_direct_loop
    _labels_raw_to_texture_direct_inner_loop = numba.njit(
        parallel=True, cache=True
    )(_labels_raw_to_texture_direct_inner_loop)
    prange = numba.prange  # type: ignore [misc]

    del numba
