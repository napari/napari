import numpy as np
import pytest

pytest.importorskip('zarr.abc', reason='requires zarr v3')
import zarr

from napari.experimental._generative_zarr import (
    MandelbrotStore,
    MandelbulbStore,
    mandelbrot,
    tile_bounds,
)

LEVELS = 4
TILESIZE = 32


@pytest.fixture
def mandelbrot_arrays():
    store = MandelbrotStore(levels=LEVELS, tilesize=TILESIZE, maxiter=255)
    group = zarr.open_group(store, mode='r')
    return [group[str(level)] for level in range(LEVELS)]


def test_mandelbrot_store_metadata(mandelbrot_arrays):
    base_width = TILESIZE * 2**LEVELS
    for level, arr in enumerate(mandelbrot_arrays):
        assert arr.shape == (base_width // 2**level,) * 2
        assert arr.chunks == (TILESIZE, TILESIZE)
        assert arr.dtype == np.uint8


def test_mandelbrot_store_multiscales_attr():
    store = MandelbrotStore(levels=LEVELS, tilesize=TILESIZE)
    group = zarr.open_group(store, mode='r')
    multiscales = group.attrs['multiscales']
    assert len(multiscales[0]['datasets']) == LEVELS


def test_mandelbrot_uint16_dtype():
    store = MandelbrotStore(levels=2, tilesize=8, maxiter=1000)
    group = zarr.open_group(store, mode='r')
    assert group['0'].dtype == np.uint16


def test_mandelbrot_chunks_are_deterministic(mandelbrot_arrays):
    arr = mandelbrot_arrays[LEVELS - 1]
    first = arr[:]
    second = arr[:]
    np.testing.assert_array_equal(first, second)
    # the mandelbrot set has both interior and escaped regions
    assert first.max() > 0
    assert first.min() == 0


def test_mandelbrot_partial_read(mandelbrot_arrays):
    arr = mandelbrot_arrays[LEVELS - 2]
    sub = arr[10:50, 20:60]
    assert sub.shape == (40, 40)
    np.testing.assert_array_equal(sub, arr[:][10:50, 20:60])


def test_mandelbrot_levels_consistent(mandelbrot_arrays):
    """Downsampling a fine level approximates the next coarser level."""
    fine = mandelbrot_arrays[LEVELS - 2][:]
    coarse = mandelbrot_arrays[LEVELS - 1][:]
    # nearest-neighbor correspondence at aligned pixels (sample, since
    # escape counts differ slightly with coordinates)
    matches = (fine[::2, ::2] == coarse).mean()
    assert matches > 0.5


def test_mandelbrot_store_through_cache():
    pytest.importorskip(
        'zarr.experimental', reason='zarr.experimental not available'
    )
    from zarr.experimental.cache_store import CacheStore
    from zarr.storage import MemoryStore

    store = MandelbrotStore(levels=LEVELS, tilesize=TILESIZE)
    cached = CacheStore(store, cache_store=MemoryStore(), max_size=int(1e8))
    group = zarr.open_group(cached, mode='r')
    direct = zarr.open_group(store, mode='r')
    np.testing.assert_array_equal(group['2'][:], direct['2'][:])


def test_mandelbulb_store():
    store = MandelbulbStore(levels=3, tilesize=8, maxiter=64)
    group = zarr.open_group(store, mode='r')
    arr = group['2']
    assert arr.shape == (16, 16, 16)
    data = arr[:]
    assert data.max() > 0


def test_tile_bounds():
    bounds = tile_bounds(0, np.array([0, 0]), max_level=0)
    np.testing.assert_allclose(bounds, [[-2.5, 2.5], [-2.5, 2.5]])
    bounds = tile_bounds(0, np.array([1, 0]), max_level=1)
    np.testing.assert_allclose(bounds, [[0.0, 2.5], [-2.5, 0.0]])


def test_mandelbrot_kernel_interior_is_maxiter():
    out = np.zeros(4 * 4, dtype=np.uint8)
    # a region inside the set: all pixels should reach maxiter
    mandelbrot(out, -0.1, -0.1, 0.1, 0.1, 4, 50)
    assert (out.reshape(4, 4) >= 49).all()


def test_unknown_keys_return_none():
    store = MandelbrotStore(levels=2, tilesize=8)
    assert store._parse_chunk_key('not-a-chunk') is None
    assert store._parse_chunk_key('5/c/0/0') is None  # level out of range
    assert store._parse_chunk_key('0/c/0') is None  # wrong ndim
    assert store._parse_chunk_key('0/c/a/b') is None  # non-integer
