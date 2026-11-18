"""Write edited label regions without accumulating Dask assignment graphs."""

from __future__ import annotations

import sys
from itertools import product
from operator import getitem
from typing import Any

import numpy as np


def _chunk_edit_task(
    source: Any,
    indices: np.ndarray,
    values: np.ndarray,
    shape: tuple[int, ...],
) -> tuple[Any, ...]:
    """Use dependency-owned callables so workers need only Dask and NumPy."""
    from dask.array.slicing import setitem

    assigned = (getitem, values, indices) if indices.dtype == bool else values
    return (
        np.reshape,
        (setitem, (np.reshape, source, (-1,)), assigned, [indices]),
        shape,
    )


def _unpack_chunk_edit(
    task: Any,
) -> tuple[Any, np.ndarray, np.ndarray] | None:
    """Recognize only the flat read-modify-write task built above."""
    from dask.array.slicing import setitem

    if not (
        isinstance(task, tuple)
        and len(task) == 3
        and task[0] is np.reshape
        and isinstance(task[1], tuple)
        and len(task[1]) == 4
        and task[1][0] is setitem
    ):
        return None
    _, source, values, index = task[1]
    if not (
        isinstance(source, tuple)
        and len(source) == 3
        and source[0] is np.reshape
        and source[2] == (-1,)
        and isinstance(index, list)
        and len(index) == 1
        and isinstance(index[0], np.ndarray)
    ):
        return None
    indices = index[0]
    if indices.dtype == bool:
        if not (
            isinstance(values, tuple)
            and len(values) == 3
            and values[0] is getitem
            and values[2] is indices
        ):
            return None
        values = values[1]
    if not isinstance(values, np.ndarray):
        return None
    return source[1], indices, values


def _merge_chunk_edit(
    task: Any,
    source_key: Any,
    indices: np.ndarray,
    values: np.ndarray,
    shape: tuple[int, ...],
) -> tuple[Any, ...]:
    """Keep only the latest value of each edited voxel in a chunk."""
    # Keep literal chunks as immutable inputs; a small edit should not copy
    # an entire large source chunk until a consumer actually reads it.
    if isinstance(task, tuple) and len(task) == 2 and task[0] is np.copy:
        source_key = task[1]
    if isinstance(task, np.ndarray):
        source_key = task

    size = int(np.prod(shape))
    previous = _unpack_chunk_edit(task)
    if previous is not None:
        source_key, old_indices, old_values = previous
        if old_indices.dtype == bool or indices.dtype == bool:
            if old_indices.dtype == bool:
                mask = old_indices.copy()
                merged = old_values.copy()
            else:
                mask = np.zeros(size, dtype=bool)
                merged = np.zeros(size, dtype=values.dtype)
                mask[old_indices] = True
                merged[old_indices] = old_values
            mask[indices] = True
            merged[indices] = (
                values[indices] if indices.dtype == bool else values
            )
            if mask.all():
                return (np.copy, merged.reshape(shape))
            return _chunk_edit_task(source_key, mask, merged, shape)

        # New values come first so overlapping edits replace older values.
        combined = np.concatenate((indices, old_indices))
        indices, positions = np.unique(combined, return_index=True)
        values = np.concatenate((values, old_values))[positions]

    if indices.dtype == bool:
        return _chunk_edit_task(source_key, indices, values, shape)
    if len(indices) == size:
        return (np.copy, values.reshape(shape))

    # Sparse indices are cheaper for a small brush on a large chunk. Once
    # dense storage is cheaper, cap the payload at one mask and one chunk.
    if indices.nbytes + values.nbytes > size * (1 + values.dtype.itemsize):
        mask = np.zeros(size, dtype=bool)
        merged = np.zeros(size, dtype=values.dtype)
        mask[indices] = True
        merged[indices] = values
        return _chunk_edit_task(source_key, mask, merged, shape)
    return _chunk_edit_task(source_key, indices, values, shape)


def _resolve_chunk_alias(
    graph: dict[Any, Any], key: tuple[Any, ...], ndim: int
) -> tuple[Any, ...]:
    """Skip the unchanged-chunk aliases introduced by basic assignment."""
    while True:
        task = graph[key]
        if (
            isinstance(task, tuple)
            and len(task) == ndim + 1
            and isinstance(task[0], str)
            and all(isinstance(index, int) for index in task[1:])
            and task in graph
        ):
            key = task
        else:
            return key


def _set_labels_data(
    data: Any, slice_key: tuple[slice, ...], values: np.ndarray | int
) -> None:
    """Assign a bounding box, coalescing ordinary Dask chunk edits.

    Dask's native assignment retains the previous graph, including an alias
    for every unchanged chunk. Repeated painting and history replay then
    make even small reads progressively more expensive. Coalesce our own
    edits per chunk and discard obsolete tasks, without computing data or
    mutating a graph retained by an earlier Dask expression.

    The optimization is limited to NumPy chunks, basic unit-step slices and
    unannotated graphs. Other backends and specialized Dask arrays retain
    their native assignment and scheduling semantics.
    """
    array_type: type[Any] | None = getattr(
        sys.modules.get('dask.array'), 'Array', None
    )
    if array_type is None or type(data) is not array_type:
        data[slice_key] = values
        return

    import dask
    from dask.array import Array
    from dask.highlevelgraph import HighLevelGraph
    from dask.optimization import cull

    # Dask 2022.1 stores active annotations in config; newer versions expose
    # get_annotations(). Flattening an annotated graph would lose constraints
    # such as resources, workers and retries used by distributed schedulers.
    get_annotations = getattr(
        dask, 'get_annotations', lambda: dask.config.get('annotations', None)
    )
    if (
        type(data._meta) is not np.ndarray
        or not isinstance(data.dask, HighLevelGraph)
        or get_annotations()
        or any(layer.annotations for layer in data.dask.layers.values())
        or any(np.isnan(size) for size in data.shape)
        or len(slice_key) != data.ndim
        or any(
            not isinstance(s, slice) or s.step not in (None, 1)
            for s in slice_key
        )
    ):
        data[slice_key] = values
        return

    bounds = [
        s.indices(n)[:2] for s, n in zip(slice_key, data.shape, strict=True)
    ]
    shape = tuple(stop - start for start, stop in bounds)
    if any(size <= 0 for size in shape):
        data[slice_key] = values
        return

    # Let Dask validate the assignment and name its new keys. Prepare on an
    # explicit shallow Array, since copy() adds a task in older Dask versions.
    # The live array is changed only after the compacted graph is complete.
    updated = Array(data.dask, data.name, data.chunks, meta=data)
    updated[slice_key] = values
    updated_name = updated.name
    # Only validation and naming need the native assignment. Release its
    # value tasks before copying the edited region into the final graph.
    del updated
    graph = dict(data.dask)
    region = np.broadcast_to(np.asarray(values, dtype=data.dtype), shape)
    boundaries = [np.concatenate(([0], np.cumsum(c))) for c in data.chunks]
    spans = [
        range(
            np.searchsorted(b, start, side='right') - 1,
            np.searchsorted(b, stop, side='left'),
        )
        for b, (start, stop) in zip(boundaries, bounds, strict=True)
    ]

    for block_index in product(*spans):
        block_shape = tuple(
            data.chunks[dim][index] for dim, index in enumerate(block_index)
        )
        ranges, region_slices = [], []
        for dim, index in enumerate(block_index):
            begin, end = boundaries[dim][index : index + 2]
            start, stop = bounds[dim]
            first, last = max(start, begin), min(stop, end)
            ranges.append(np.arange(first - begin, last - begin))
            region_slices.append(slice(first - start, last - start))
        assigned = np.array(
            region[tuple(region_slices)], copy=True, order='C'
        ).reshape(-1)
        if assigned.size == np.prod(block_shape):
            # Full overwrites need neither source data nor a voxel index.
            # Copy on compute: old Dask exposes single-chunk results directly
            # and Labels must not mutate a graph-owned buffer via its cache.
            graph[(updated_name, *block_index)] = (
                np.copy,
                assigned.reshape(block_shape),
            )
            continue
        size = int(np.prod(block_shape))
        if assigned.size * (
            np.dtype(np.intp).itemsize + data.dtype.itemsize
        ) > size * (1 + data.dtype.itemsize):
            # Large partial writes also use a mask directly, avoiding a
            # temporary integer index many times larger than uint8 labels.
            indices = np.zeros(size, dtype=bool)
            chunk_region = tuple(slice(r[0], r[-1] + 1) for r in ranges)
            indices.reshape(block_shape)[chunk_region] = True
            dense_values = np.zeros(size, dtype=data.dtype)
            dense_values.reshape(block_shape)[chunk_region] = assigned.reshape(
                tuple(len(r) for r in ranges)
            )
            assigned = dense_values
        else:
            indices = np.ravel_multi_index(
                np.ix_(*ranges), block_shape
            ).ravel()
        source_key = _resolve_chunk_alias(
            graph, (data.name, *block_index), data.ndim
        )
        graph[(updated_name, *block_index)] = _merge_chunk_edit(
            graph[source_key], source_key, indices, assigned, block_shape
        )

    keys = list(
        product([updated_name], *(range(count) for count in data.numblocks))
    )
    for key in keys:
        if key not in graph:
            source_key = _resolve_chunk_alias(
                graph, (data.name, *key[1:]), data.ndim
            )
            source = graph[source_key]
            graph[key] = (
                source if isinstance(source, np.ndarray) else source_key
            )
    graph, _ = cull(graph, keys)
    compacted = HighLevelGraph({updated_name: graph}, {updated_name: set()})
    # Match native Array.__setitem__: preserve identity and invalidate cached
    # output keys through the _name setter. Shape, chunks and dtype are stable.
    data.dask = compacted
    data._name = updated_name
