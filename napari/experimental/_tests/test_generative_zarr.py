import numpy as np
import pytest

from napari.experimental._generative_zarr import (
    MandelbrotStore,
    MandelbulbStore,
    create_meta_store,
    tile_bounds,
)


def test_create_meta_store():
    levels = 2
    tilesize = 512
    compressor = None
    dtype = float
    ndim = 3

    store = create_meta_store(levels, tilesize, compressor, dtype, ndim=ndim)

    assert '.zgroup' in store
    assert len(store) == levels + 2


def test_tile_bounds():
    level = 2
    coords = [10, 20]
    max_level = 3
    min_coord = -1
    max_coord = 4
    bounds = tile_bounds(
        level,
        coords,
        max_level,
        min_coord=min_coord,
        max_coord=max_coord,
    )
    assert isinstance(bounds, np.ndarray)
    assert np.all(bounds == np.array([[24, 26.5], [49.0, 51.5]]))


@pytest.mark.parametrize('max_level', [8, 14])
def test_MandlebrotStore(max_level):
    store = MandelbrotStore(
        levels=max_level,
        tilesize=512,
        compressor=None,
        maxiter=255,
    )
    assert (
        len(store.listdir()) == max_level + 2
    )  # number of levels + .zattrs and .zgroup


@pytest.mark.parametrize('max_level', [8, 14])
def test_MandelbulbStore(max_level):
    store = MandelbulbStore(
        levels=max_level,
        tilesize=32,
        compressor=None,
        maxiter=255,
    )
    assert (
        len(store.listdir()) == max_level + 2
    )  # number of levels + .zattrs and .zgroup
