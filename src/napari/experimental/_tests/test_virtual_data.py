import threading

import dask.array as da
import numpy as np
import pytest

from napari.experimental._virtual_data import (
    MultiScaleVirtualData,
    VirtualData,
    chunk_boundaries,
    chunk_shape_for,
    chunk_sizes_for,
)


@pytest.fixture
def base_array():
    return np.arange(100 * 120, dtype=np.uint16).reshape(100, 120)


@pytest.fixture
def dask_array(base_array):
    return da.from_array(base_array, chunks=(32, 40))


def test_chunk_shape_for_dask(dask_array):
    assert chunk_shape_for(dask_array) == (32, 40)


def test_chunk_shape_for_numpy(base_array):
    shape = chunk_shape_for(base_array)
    assert len(shape) == 2
    assert all(s > 0 for s in shape)


def test_chunk_shape_for_zarr():
    zarr = pytest.importorskip('zarr')
    arr = zarr.zeros((100, 120), chunks=(32, 40), dtype='u2')
    assert chunk_shape_for(arr) == (32, 40)


def test_chunk_boundaries_dask(dask_array):
    bounds = chunk_boundaries(dask_array)
    np.testing.assert_array_equal(bounds[0], [0, 32, 64, 96, 100])
    np.testing.assert_array_equal(bounds[1], [0, 40, 80, 120])


class _RectilinearArrayStub:
    """Mimics a zarr array with a rectilinear chunk grid.

    Such arrays expose per-dimension chunk sizes via ``read_chunk_sizes``
    and raise ``NotImplementedError`` from ``.chunks`` (only defined for
    regular grids) — so ``read_chunk_sizes`` must be consulted first.
    """

    shape = (10, 20)
    dtype = np.dtype('u1')
    read_chunk_sizes = ((2, 3, 5), (10, 10))

    @property
    def chunks(self):
        raise NotImplementedError('chunk grid is not regular')


def test_chunk_helpers_rectilinear_stub():
    arr = _RectilinearArrayStub()
    assert chunk_sizes_for(arr) == ((2, 3, 5), (10, 10))
    assert chunk_shape_for(arr) == (5, 10)
    bounds = chunk_boundaries(arr)
    np.testing.assert_array_equal(bounds[0], [0, 2, 5, 10])
    np.testing.assert_array_equal(bounds[1], [0, 10, 20])


def test_virtual_data_rectilinear_alignment():
    vdata = VirtualData(_RectilinearArrayStub())
    # intervals snap outward to the irregular chunk edges
    assert vdata.chunk_aligned_interval((1, 0), (4, 5)) == ([0, 0], [5, 10])
    assert vdata.chunk_aligned_interval((6, 12), (9, 15)) == (
        [5, 10],
        [10, 20],
    )


def test_chunk_helpers_zarr_rectilinear():
    """Exercise a real rectilinear-chunked zarr array when supported."""
    zarr = pytest.importorskip('zarr')
    try:
        with zarr.config.set({'array.rectilinear_chunks': True}):
            arr = zarr.create_array(
                {},
                shape=(10, 20),
                chunks=((2, 3, 5), (10, 10)),
                dtype='u1',
            )
    except (TypeError, ValueError):
        pytest.skip('installed zarr lacks rectilinear chunk grids')
    assert chunk_sizes_for(arr) == ((2, 3, 5), (10, 10))
    bounds = chunk_boundaries(arr)
    np.testing.assert_array_equal(bounds[0], [0, 2, 5, 10])
    np.testing.assert_array_equal(bounds[1], [0, 10, 20])


def test_virtual_data_protocol(dask_array):
    """VirtualData satisfies napari's LayerDataProtocol."""
    from napari.layers._data_protocols import assert_protocol

    vdata = VirtualData(dask_array)
    assert_protocol(vdata)
    assert vdata.shape == (100, 120)
    assert vdata.ndim == 2
    assert vdata.dtype == np.uint16
    assert vdata.size == 100 * 120


def test_set_interval_chunk_aligned(dask_array):
    vdata = VirtualData(dask_array)
    vdata.set_interval((10, 50), (60, 70))
    assert vdata.interval == ((0, 40), (64, 80))
    assert vdata.hyperslice.shape == (64, 40)
    assert vdata.translate == (0, 40)


def test_set_interval_clamps_to_shape(dask_array):
    vdata = VirtualData(dask_array)
    vdata.set_interval((-10, -10), (1000, 1000))
    assert vdata.interval == ((0, 0), (100, 120))


def test_reads_outside_interval_are_zero(dask_array, base_array):
    vdata = VirtualData(dask_array)
    vdata.set_interval((32, 40), (64, 80))
    vdata.set_offset((slice(32, 64), slice(40, 80)), base_array[32:64, 40:80])

    result = np.asarray(vdata[0:100, 0:120])
    assert result.shape == (100, 120)
    np.testing.assert_array_equal(
        result[32:64, 40:80],
        base_array[32:64, 40:80],
    )
    assert result[:32].sum() == 0
    assert result[64:].sum() == 0


def test_getitem_composes_like_napari_slicing(dask_array, base_array):
    """Napari slices displayed dims first, then indexes the point dims."""
    vdata = VirtualData(dask_array)
    vdata.set_interval((0, 0), (100, 120))
    vdata.set_offset((slice(0, 100), slice(0, 120)), base_array)

    # First the displayed-axis crop, then a point selection on the result.
    view = vdata[(slice(None), slice(20, 70))]
    point_view = view[(40,)]
    np.testing.assert_array_equal(
        np.asarray(point_view),
        base_array[40, 20:70],
    )


def test_getitem_int_indexing(dask_array, base_array):
    vdata = VirtualData(dask_array)
    vdata.set_interval((32, 40), (64, 80))
    vdata.set_offset((slice(32, 64), slice(40, 80)), base_array[32:64, 40:80])

    assert np.asarray(vdata[40][50]) == base_array[40, 50]
    # outside the interval -> zero
    assert np.asarray(vdata[0][0]) == 0


def test_getitem_rejects_unsupported_keys(dask_array):
    vdata = VirtualData(dask_array)
    with pytest.raises(IndexError):
        vdata[::2]
    with pytest.raises(IndexError):
        vdata[0, 0, 0]


def test_set_interval_preserves_overlap(dask_array, base_array):
    vdata = VirtualData(dask_array)
    vdata.set_interval((0, 0), (64, 80))
    vdata.set_offset((slice(0, 64), slice(0, 80)), base_array[:64, :80])

    vdata.set_interval((32, 40), (100, 120))
    result = np.asarray(vdata[32:64, 40:80])
    np.testing.assert_array_equal(result, base_array[32:64, 40:80])


def test_set_interval_prunes_loaded_chunks(dask_array):
    vdata = VirtualData(dask_array)
    vdata.set_interval((0, 0), (64, 80))
    vdata.loaded_chunks.add(((0, 32), (0, 40)))
    vdata.loaded_chunks.add(((32, 64), (40, 80)))

    vdata.set_interval((32, 40), (100, 120))
    assert vdata.loaded_chunks == {((32, 64), (40, 80))}


def test_set_offset_clips_to_interval(dask_array, base_array):
    vdata = VirtualData(dask_array)
    vdata.set_interval((32, 40), (64, 80))
    # write overlaps the interval only partially
    vdata.set_offset((slice(0, 64), slice(0, 80)), base_array[:64, :80])
    np.testing.assert_array_equal(
        np.asarray(vdata[32:64, 40:80]),
        base_array[32:64, 40:80],
    )


def test_set_offset_outside_interval_is_ignored(dask_array, base_array):
    vdata = VirtualData(dask_array)
    vdata.set_interval((64, 80), (100, 120))
    vdata.set_offset((slice(0, 32), slice(0, 40)), base_array[:32, :40])
    assert vdata.hyperslice.sum() == 0


def test_concurrent_reads_and_writes(dask_array, base_array):
    """Concurrent set_offset/set_interval/reads must not raise."""
    vdata = VirtualData(dask_array)
    vdata.set_interval((0, 0), (64, 80))
    errors = []

    def writer():
        try:
            for _ in range(50):
                vdata.set_offset(
                    (slice(0, 32), slice(0, 40)),
                    base_array[:32, :40],
                )
        except Exception as e:  # pragma: no cover  # noqa: BLE001
            errors.append(e)

    def mover():
        try:
            for i in range(50):
                offset = (i % 2) * 32
                vdata.set_interval(
                    (offset, offset),
                    (offset + 64, offset + 80),
                )
        except Exception as e:  # pragma: no cover  # noqa: BLE001
            errors.append(e)

    def reader():
        try:
            for _ in range(50):
                np.asarray(vdata[0:100, 0:120])
        except Exception as e:  # pragma: no cover  # noqa: BLE001
            errors.append(e)

    threads = [
        threading.Thread(target=t) for t in (writer, mover, reader, reader)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_multiscale_virtual_data(base_array):
    coarse = base_array[::2, ::2].copy()
    msvd = MultiScaleVirtualData([base_array, coarse])
    assert len(msvd) == 2
    assert msvd.shape == (100, 120)
    assert msvd._scale_factors[0] == [1.0, 1.0]
    assert msvd._scale_factors[1] == [2.0, 2.0]


def test_multiscale_virtual_data_empty():
    with pytest.raises(ValueError, match='non-empty'):
        MultiScaleVirtualData([])


def test_multiscale_backdrop(base_array):
    coarse = base_array[::2, ::2].copy()
    msvd = MultiScaleVirtualData([base_array, coarse])

    # make the coarse level resident and loaded
    msvd[1].set_interval((0, 0), coarse.shape)
    msvd[1].set_offset(
        (slice(0, coarse.shape[0]), slice(0, coarse.shape[1])),
        coarse,
    )

    msvd.set_interval(0, (20, 20), (60, 60), backdrop_level=1)
    vdata = msvd[0]
    min_coord, _max_coord = vdata.interval
    # every value should be nearest-neighbor upsampled from the coarse level
    sample = vdata.hyperslice[5, 7]
    abs_y, abs_x = min_coord[0] + 5, min_coord[1] + 7
    assert sample == coarse[abs_y // 2, abs_x // 2]


def test_multiscale_backdrop_without_resident_source(base_array):
    coarse = base_array[::2, ::2].copy()
    msvd = MultiScaleVirtualData([base_array, coarse])
    # no interval set on the coarse level: backdrop falls back to zeros
    msvd.set_interval(0, (20, 20), (60, 60), backdrop_level=1)
    assert msvd[0].hyperslice.sum() == 0


def test_multiscale_data_wrapper_accepts_virtual_data(dask_array):
    from napari.layers._multiscale_data import MultiScaleData

    msvd = MultiScaleVirtualData([dask_array, dask_array[::2, ::2]])
    wrapped = MultiScaleData(msvd._data)
    assert wrapped.shape == (100, 120)
    assert wrapped.dtype == np.uint16
