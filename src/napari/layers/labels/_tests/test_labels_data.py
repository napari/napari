import subprocess
import sys

import cloudpickle
import dask
import dask.array as da
import numpy as np
import pytest
from dask.cache import Cache

from napari.layers import Labels
from napari.layers.labels._labels_data import (
    _set_labels_data,
    _unpack_chunk_edit,
)
from napari.utils import _dask_utils


@pytest.mark.parametrize(
    ('shape', 'chunks'),
    [
        ((17, 19), (6, 7)),
        ((11, 13, 15), (4, 5, 6)),
        ((2, 9, 11, 13), (1, 4, 5, 6)),
    ],
)
@pytest.mark.parametrize('lazy', [False, True])
def test_overlapping_chunk_edits(shape, chunks, lazy):
    rng = np.random.default_rng(9913)
    expected = rng.integers(0, 10, shape, dtype=np.uint32)
    original = expected.copy()
    data = da.from_array(expected.copy(), chunks=chunks)
    if lazy:
        data = data + 1
        expected += 1
        original += 1
    snapshot = data.copy()

    for _ in range(40):
        starts = [rng.integers(0, size) for size in shape]
        stops = [
            rng.integers(start + 1, size + 1)
            for start, size in zip(starts, shape, strict=True)
        ]
        key = tuple(
            slice(start, stop)
            for start, stop in zip(starts, stops, strict=True)
        )
        values = rng.integers(0, 20, expected[key].shape, dtype=expected.dtype)
        expected[key] = values
        _set_labels_data(data, key, values)
        values.fill(0)  # The caller can reuse its buffer immediately.
        np.testing.assert_array_equal(data.compute(), expected)

    np.testing.assert_array_equal(snapshot.compute(), original)


@pytest.mark.parametrize('chunks', [(32, 32), (7, 9)])
def test_repeated_edits_have_bounded_graph_and_payload(chunks):
    data = da.zeros((32, 32), chunks=chunks, dtype=np.uint8)
    key = (slice(2, 5), slice(3, 6))
    _set_labels_data(data, key, 1)
    graph_size = len(data.dask)
    for value in range(2, 62):
        _set_labels_data(data, key, value)
        assert len(data.dask) == graph_size

    expected = np.zeros(data.shape, dtype=data.dtype)
    expected[key] = 61
    np.testing.assert_array_equal(data.compute(), expected)
    # A long stroke retains current values, not a list of previous writes.
    edits = [_unpack_chunk_edit(task) for task in data.dask.values()]
    payload = sum(
        edit[1].nbytes + edit[2].nbytes for edit in edits if edit is not None
    )
    assert payload <= 9 * (np.dtype(np.intp).itemsize + data.dtype.itemsize)


def test_dense_chunk_edits_and_snapshots():
    data = da.zeros((8, 8), chunks=(8, 8), dtype=np.uint8)
    expected = np.zeros(data.shape, dtype=data.dtype)
    snapshots = []
    for row in range(8):
        key = (slice(row, row + 1), slice(None))
        _set_labels_data(data, key, row + 1)
        expected[key] = row + 1
        snapshots.append((data.copy(), expected.copy()))
        for task in data.dask.values():
            edit = _unpack_chunk_edit(task)
            if edit is not None:
                assert edit[1].nbytes + edit[2].nbytes <= 2 * expected.nbytes
    assert len(data.dask) == 1  # A fully overwritten chunk needs no source.
    for snapshot, values in snapshots:
        np.testing.assert_array_equal(snapshot.compute(), values)


@pytest.mark.parametrize('stop', [63, 64], ids=['large-partial', 'full'])
def test_large_writes_do_not_allocate_voxel_indices(stop, monkeypatch):
    def unexpected_indices(*args, **kwargs):
        raise AssertionError(
            'Large writes should not allocate integer voxel indices'
        )

    monkeypatch.setattr(np, 'ravel_multi_index', unexpected_indices)
    data = da.zeros((64, 64), chunks=(64, 64), dtype=np.uint8)
    for value in (3, 7):
        _set_labels_data(data, (slice(0, stop), slice(0, stop)), value)
    expected = np.zeros(data.shape, dtype=data.dtype)
    expected[:stop, :stop] = 7
    np.testing.assert_array_equal(data.compute(), expected)


def test_edits_remain_lazy():
    reads = []
    original = np.ones((4, 4), dtype=np.uint8)

    @dask.delayed
    def read_block(index):
        reads.append(index)
        return original

    data = da.concatenate(
        [
            da.from_delayed(read_block(i), shape=(4, 4), dtype=np.uint8)
            for i in range(2)
        ]
    )
    key = (slice(1, 3), slice(1, 3))
    for value in range(2, 12):
        _set_labels_data(data, key, value)
    assert reads == []
    with dask.config.set(scheduler='synchronous'):
        np.testing.assert_array_equal(data[key].compute(), 11)
    assert reads == [0]
    np.testing.assert_array_equal(original, 1)


@pytest.mark.parametrize('stop', [2, 7, 8], ids=['sparse', 'dense', 'full'])
def test_edited_graph_computes_without_napari_on_worker(stop):
    data = da.ones((8, 8), chunks=(8, 8), dtype=np.uint8)
    _set_labels_data(data, (slice(0, stop), slice(0, stop)), 7)
    script = """
import sys
class NoNapari:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'napari' or fullname.startswith('napari.'):
            raise ModuleNotFoundError('napari is not installed on this worker')
sys.meta_path.insert(0, NoNapari())
import cloudpickle
import numpy as np
array, stop = cloudpickle.loads(sys.stdin.buffer.read())
expected = np.ones((8, 8), dtype=np.uint8)
expected[:stop, :stop] = 7
np.testing.assert_array_equal(array.compute(scheduler='synchronous'), expected)
"""
    result = subprocess.run(
        [sys.executable, '-c', script],
        input=cloudpickle.dumps((data, stop)),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()


def test_edits_preserve_references_snapshots_and_cache(monkeypatch):
    monkeypatch.setattr(_dask_utils, '_DASK_CACHE', Cache(2**20))
    data = da.zeros((11, 13), chunks=(4, 5), dtype=np.uint16)
    reference = data
    sliced = data[1:8, 2:9]
    derived = data + 1
    data.__dask_keys__()  # Exercise invalidation of Dask's cached output keys.
    with _dask_utils.configure_dask(data)():
        np.testing.assert_array_equal(data.compute(), 0)
        _set_labels_data(data, (slice(1, 8), slice(2, 9)), 7)
        assert data is reference
        np.testing.assert_array_equal(reference[1:8, 2:9].compute(), 7)
        # An assignment made outside Labels must also survive the next edit.
        data[2:5, 3:7] = 9
        _set_labels_data(data, (slice(3, 4), slice(4, 5)), 11)
        expected = np.zeros(data.shape, dtype=data.dtype)
        expected[1:8, 2:9] = 7
        expected[2:5, 3:7] = 9
        expected[3:4, 4:5] = 11
        np.testing.assert_array_equal(data.compute(), expected)
        np.testing.assert_array_equal(sliced.compute(), 0)
        np.testing.assert_array_equal(derived.compute(), 1)


@pytest.mark.parametrize('annotate_source', [False, True])
def test_scheduler_annotations_are_preserved(annotate_source):
    if annotate_source:
        with dask.annotate(resources={'GPU': 1}, retries=2):
            data = da.ones((8, 8), chunks=(4, 4), dtype=np.uint8)
        _set_labels_data(data, (slice(1, 3), slice(1, 3)), 5)
    else:
        data = da.ones((8, 8), chunks=(4, 4), dtype=np.uint8)
        with dask.annotate(resources={'GPU': 1}, retries=2):
            _set_labels_data(data, (slice(1, 3), slice(1, 3)), 5)
    assert any(
        layer.annotations == {'resources': {'GPU': 1}, 'retries': 2}
        for layer in data.dask.layers.values()
    )
    np.testing.assert_array_equal(data[1:3, 1:3].compute(), 5)


def test_specialized_arrays_use_native_assignment():
    class CustomArray(da.Array):
        assigned = False

        def __setitem__(self, key, value):
            self.assigned = True
            super().__setitem__(key, value)

    original = da.zeros((4, 4), chunks=(2, 2), dtype=np.uint8)
    data = CustomArray(
        original.dask, original.name, original.chunks, meta=original
    )
    _set_labels_data(data, (slice(1, 3), slice(1, 3)), 4)
    assert data.assigned
    np.testing.assert_array_equal(data[1:3, 1:3].compute(), 4)

    masked = np.ma.array(np.ones((4, 4), dtype=np.uint8), mask=False)
    masked.mask[0, 0] = True
    data = da.from_array(masked, chunks=(2, 2))
    _set_labels_data(data, (slice(1, 3), slice(1, 3)), 4)
    result = data.compute()
    assert isinstance(result, np.ma.MaskedArray)
    np.testing.assert_array_equal(result.mask, masked.mask)
    np.testing.assert_array_equal(result[1:3, 1:3], 4)


def test_invalid_assignment_leaves_array_unchanged():
    data = da.zeros((8, 8), chunks=(4, 4), dtype=np.uint8)
    graph, name = data.dask, data.name
    with pytest.raises(ValueError, match='shape mismatch'):
        _set_labels_data(data, (slice(1, 3), slice(1, 3)), np.ones((3, 3)))
    assert data.dask is graph
    assert data.name == name


@pytest.mark.parametrize('cache', [False, True])
def test_snapshot_after_undo_is_not_mutated_by_display_cache(
    cache, monkeypatch
):
    monkeypatch.setattr(_dask_utils, '_DASK_CACHE', Cache(2**20))
    data = da.from_array(np.zeros((16, 16), dtype=np.uint8), chunks=(16, 16))
    layer = Labels(data, cache=cache)
    layer.brush_size = 3
    layer.fill((3, 3), 2)
    layer.undo()  # Refresh loads a fully overwritten chunk into the raw cache.
    snapshot = data.copy()
    expected = np.asarray(data).copy()
    layer.paint((8, 8), 5)
    assert np.any(np.asarray(data) == 5)
    np.testing.assert_array_equal(np.asarray(snapshot), expected)


@pytest.mark.parametrize('chunks', [(17, 19), (6, 7)])
@pytest.mark.parametrize('preserve_labels', [False, True])
def test_labels_edit_history_and_abort(chunks, preserve_labels, monkeypatch):
    # Older Dask can return cached arrays by reference. Isolate each scenario
    # from display-cache mutations made by earlier, identically named data.
    monkeypatch.setattr(_dask_utils, '_DASK_CACHE', Cache(2**20))
    original = np.zeros((17, 19), dtype=np.uint32)
    original[3:12, 4:14] = 1
    source = da.from_array(original, chunks=chunks) + 0
    layer = Labels(source)
    reference = Labels(original.copy())
    for target in (layer, reference):
        target.brush_size = 5
        target.preserve_labels = preserve_labels
        with target.block_history():
            for coordinate in [(3, 4), (5, 7), (7, 10), (12, 13)]:
                target.paint(coordinate, 2)
        target.fill((5, 7), 3)
        target.paint_polygon(np.array([[2, 2], [8, 3], [6, 12]]), 4)
    assert layer.data is source

    def compare():
        np.testing.assert_array_equal(np.asarray(layer.data), reference.data)
        np.testing.assert_array_equal(
            layer._slice.image.raw, reference._slice.image.raw
        )
        assert len(layer._undo_history) == len(reference._undo_history)
        assert len(layer._redo_history) == len(reference._redo_history)

    compare()
    for action in ['undo'] * 3 + ['redo'] * 3:
        getattr(layer, action)()
        getattr(reference, action)()
        compare()
    for target in (layer, reference):
        target._begin_stroke()
        target.paint((7, 8), 0)
        target.paint((11, 12), 0)
    compare()
    for target in (layer, reference):
        target._abort_stroke()
    compare()
