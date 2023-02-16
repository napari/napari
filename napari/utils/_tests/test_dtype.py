import itertools

import numpy as np
import pytest
import zarr
from dask import array as da

from napari.utils._dtype import get_dtype_limits, normalize_dtype

ts = pytest.importorskip('tensorstore')
torch = pytest.importorskip('torch')

bit_depths = [str(2**i) for i in range(3, 7)]
uints = ['uint' + b for b in bit_depths]
ints = ['int' + b for b in bit_depths]
floats = ['float32', 'float64']
complex_types = ['complex64', 'complex128']
bools = ['bool']
pure_py = ['int', 'float']


@pytest.mark.parametrize(
    'dtype_str', ['uint8'] + ints + floats + complex_types + bools
)
def test_normalize_dtype_torch(dtype_str):
    """torch doesn't have uint for >8bit, so it gets its own test."""
    # torch doesn't let you specify dtypes as str,
    # see https://github.com/pytorch/pytorch/issues/40568
    torch_arr = torch.zeros(5, dtype=getattr(torch, dtype_str))
    np_arr = np.zeros(5, dtype=dtype_str)
    assert normalize_dtype(torch_arr.dtype) is np_arr.dtype.type


@pytest.mark.parametrize(
    'dtype_str', uints + ints + floats + complex_types + bools
)
def test_normalize_dtype_tensorstore(dtype_str):
    np_arr = np.zeros(5, dtype=dtype_str)
    ts_arr = ts.array(np_arr)  # inherit ts dtype from np dtype
    assert normalize_dtype(ts_arr.dtype) is np_arr.dtype.type


@pytest.mark.parametrize(
    'module, dtype_str',
    itertools.product(
        (np, da, zarr), uints + ints + floats + complex_types + bools
    ),
)
def test_normalize_dtype_np_noop(module, dtype_str):
    """Check that normalize dtype works as expected for plain NumPy dtypes."""
    module_arr = module.zeros(5, dtype=dtype_str)
    np_arr = np.zeros(5, dtype=normalize_dtype(module_arr.dtype))
    assert normalize_dtype(module_arr.dtype) is normalize_dtype(np_arr.dtype)


@pytest.mark.parametrize('dtype_str', ['int', 'float'])
def test_pure_python_types(dtype_str):
    pure_arr = np.zeros(5, dtype=dtype_str)
    norm_arr = np.zeros(5, dtype=normalize_dtype(dtype_str))
    assert pure_arr.dtype is norm_arr.dtype


# note: we don't write specific tests for zarr and dask because they use numpy
# dtypes directly.


@pytest.mark.parametrize(
    "dtype",
    [
        int,
        'uint8',
        np.uint8,
        'int8',
        'uint16',
        'int16',
        'uint32',
        'int32',
        float,
        'float32',
        'float64',
        '>f4',
        '>f8',
    ]
    + [''.join(t) for t in itertools.product('<>', 'iu', '1248')],
)
def test_dtype_lims(dtype):
    lims = get_dtype_limits(dtype)
    assert isinstance(lims, tuple) and len(lims) == 2
