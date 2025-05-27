from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

_np_uints: Final[dict[int, np.dtype]] = {
    8: np.dtype(np.uint8),
    16: np.dtype(np.uint16),
    32: np.dtype(np.uint32),
    64: np.dtype(np.uint64),
}

_np_ints: Final[dict[int, np.dtype]] = {
    8: np.dtype(np.int8),
    16: np.dtype(np.int16),
    32: np.dtype(np.int32),
    64: np.dtype(np.int64),
}

_np_floats: Final[dict[int, np.dtype]] = {
    16: np.dtype(np.float16),
    32: np.dtype(np.float32),
    64: np.dtype(np.float64),
}

_np_complex: Final[dict[int, np.dtype]] = {
    64: np.dtype(np.complex64),
    128: np.dtype(np.complex128),
}

_np_kinds: Final[dict[str, dict[int, np.dtype]]] = {
    'uint': _np_uints,
    'int': _np_ints,
    'float': _np_floats,
    'complex': _np_complex,
}


def _normalize_str_by_bit_depth(dtype_str: str, kind: str) -> np.dtype:
    if not any(str.isdigit(c) for c in dtype_str):  # Python 'int' or 'float'
        return np.dtype(kind)
    bit_dict = _np_kinds[kind]
    if '128' in dtype_str:
        return bit_dict[128]
    if '8' in dtype_str:
        return bit_dict[8]
    if '16' in dtype_str:
        return bit_dict[16]
    if '32' in dtype_str:
        return bit_dict[32]
    if '64' in dtype_str:
        return bit_dict[64]
    raise ValueError(f'Unrecognized bit depth in dtype: {dtype_str}')


def normalize_dtype(dtype_spec: 'DTypeLike') -> np.dtype:
    """Return a proper NumPy type given ~any duck array dtype.

    Parameters
    ----------
    dtype_spec : numpy dtype, numpy type, torch dtype, tensorstore dtype, etc
        A type that can be interpreted as a NumPy numeric data type, e.g.
        'uint32', np.uint8, torch.float32, etc.

    Returns
    -------
    dtype : numpy.dtype
        The corresponding dtype.

    Notes
    -----
    half-precision floats are not supported.
    """
    if isinstance(dtype_spec, np.dtype):
        # Handle NumPy dtypes directly, especially big endian types.
        # FIXME: handle big endian types from tensorstore
        return dtype_spec
    if isinstance(dtype_spec, str):
        return np.dtype(dtype_spec)
    dtype_str = str(dtype_spec)
    if 'uint' in dtype_str:
        return _normalize_str_by_bit_depth(dtype_str, 'uint')
    if 'int' in dtype_str:
        return _normalize_str_by_bit_depth(dtype_str, 'int')
    if 'float' in dtype_str:
        return _normalize_str_by_bit_depth(dtype_str, 'float')
    if 'complex' in dtype_str:
        return _normalize_str_by_bit_depth(dtype_str, 'complex')
    if 'bool' in dtype_str:
        return np.dtype(np.bool_)

    raise ValueError(f'Unrecognized dtype: {dtype_spec}')


def get_dtype_limits(dtype_spec: 'DTypeLike') -> tuple[float, float]:
    """Return machine limits for numeric types.

    Parameters
    ----------
    dtype_spec : numpy dtype, numpy type, torch dtype, tensorstore dtype, etc
        A type that can be interpreted as a NumPy numeric data type, e.g.
        'uint32', np.uint8, torch.float32, etc.

    Returns
    -------
    limits : tuple
        The smallest/largest numbers expressible by the type.
    """
    dtype = normalize_dtype(dtype_spec)
    info: np.iinfo | np.finfo
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
    elif dtype and np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
    else:
        raise TypeError(f'Unrecognized or non-numeric dtype: {dtype_spec}')
    return float(info.min), float(info.max)


vispy_texture_dtype = np.float32
