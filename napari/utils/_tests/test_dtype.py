import numpy as np
import pytest
import tensorstore as ts
import torch

from napari.utils._dtype import normalize_dtype

bit_depths = [str(2 ** i) for i in range(3, 7)]
uints = ['uint' + b for b in bit_depths]
ints = ['int' + b for b in bit_depths]
floats = ['float32', 'float64']
complex = ['complex64', 'complex128']
bools = ['bool']


@pytest.mark.parametrize(
    'dtype_str', ['uint8'] + ints + floats + complex + bools
)
def test_normalize_dtype_torch(dtype_str):
    """torch doesn't have uint for >8bit, so it gets its own test."""
    # torch doesn't let you specify dtypes as str,
    # see https://github.com/pytorch/pytorch/issues/40568
    torch_arr = torch.zeros(5, dtype=getattr(torch, dtype_str))
    np_arr = np.zeros(5, dtype=dtype_str)
    assert normalize_dtype(torch_arr.dtype) is np_arr.dtype.type


@pytest.mark.parametrize('dtype_str', uints + ints + floats + complex + bools)
def test_normalize_dtype_tensorstore(dtype_str):
    np_arr = np.zeros(5, dtype=dtype_str)
    ts_arr = ts.array(np_arr)  # inherit ts dtype from np dtype
    assert normalize_dtype(ts_arr.dtype) is np_arr.dtype.type


@pytest.mark.parametrize('dtype_str', uints + ints + floats + complex + bools)
def test_normalize_dtype_np_noop(dtype_str):
    """Check that normalize dtype works as expected for plain NumPy."""
    np_arr = np.zeros(5, dtype=dtype_str)
    np_arr2 = np.zeros(5, dtype=normalize_dtype(np_arr.dtype))
    assert normalize_dtype(np_arr) is normalize_dtype(np_arr2)


# note: we don't write specific tests for zarr and dask because they use numpy
# dtypes directly.
