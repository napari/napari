import itertools

import numpy as np
import pytest

pytest.importorskip('zarr')

from napari.experimental._progressive_loading_datasets import (
    mandelbrot_dataset,
    mandelbulb_dataset,
)


def test_mandelbrot_dataset():
    dataset = mandelbrot_dataset(max_levels=4, tilesize=32)
    arrays = dataset['arrays']
    assert len(arrays) == 4
    assert arrays[0].shape == (32 * 2**4,) * 2
    for fine, coarse in itertools.pairwise(arrays):
        assert fine.shape[0] == coarse.shape[0] * 2
    # data is readable and nontrivial
    coarsest = arrays[-1][:]
    assert coarsest.max() > 0


def test_mandelbrot_dataset_cache_consistency():
    dataset = mandelbrot_dataset(max_levels=3, tilesize=16)
    arr = dataset['arrays'][1]
    np.testing.assert_array_equal(arr[:], arr[:])


def test_mandelbulb_dataset():
    dataset = mandelbulb_dataset(max_levels=3, tilesize=8, maxiter=32)
    arrays = dataset['arrays']
    assert len(arrays) == 3
    assert arrays[0].ndim == 3
    assert arrays[-1][:].max() > 0
