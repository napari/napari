import pytest

from napari.experimental._generative_zarr import MandelbrotStore, \
    create_meta_store, tile_bounds, mandelbrot, hypercomplex_exponentiation, \
    mandelbulb, MandelbulbStore


@pytest.mark.parametrize('max_level', [8, 14])
def test_MandlebrotStore(max_level):
    store = MandelbrotStore(
        levels=max_level, 
        tilesize=512, 
        compressor=None, 
        maxiter=255
    )
    assert len(store.listdir()) == max_level + 2  # number of levels + .zattrs and .zgroup)


@pytest.mark.parametrize('max_level', [8, 14])
def test_MandelbulbStore(max_level):
    store = MandelbulbStore(
            levels=max_level,
            tilesize=32,
            compressor=None,
            maxiter=255
        )
    assert len(store.listdir()) == max_level + 2  # number of levels + .zattrs and .zgroup)
