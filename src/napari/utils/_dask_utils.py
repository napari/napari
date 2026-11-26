"""Dask cache utilities."""

import atexit
import collections.abc
import contextlib
import os
from collections.abc import Callable, Generator, Hashable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Lock
from typing import Any

import cachey
import dask
import dask.threaded
import numpy as np
from dask.base import get_scheduler, named_schedulers
from dask.cache import Cache
from dask.callbacks import Callback
from dask.local import get_async

try:
    from dask.threaded import ContextAwareThreadPoolExecutor as _PoolExecutor
except ImportError:  # older dask
    _PoolExecutor = ThreadPoolExecutor

_CACHE_LOCK = Lock()


class _DaskCacheStorage(cachey.Cache):
    def put(
        self,
        key: Hashable,
        value: Any,
        cost: float,
        nbytes: int | None = None,
    ) -> None:
        with _CACHE_LOCK:
            # Concurrent graphs may finish the same task before either can
            # reuse it. cachey counts a replacement as another allocation.
            if key not in self.data:
                super().put(key, value, cost, nbytes)


class _DaskSliceCache(Cache):
    """Keep task timings per computation while sharing cached results."""

    def _start(self, dsk: dict[Hashable, Any]) -> None:
        with _CACHE_LOCK:
            super()._start(dsk)


#: Shared results for opportunistic caching, sized by resize_dask_cache.
#: Individual layers can opt out using Layer(..., cache=False).
_DASK_CACHE = _DaskSliceCache(_DaskCacheStorage(1))
_DEFAULT_MEM_FRACTION = 0.25

_FUSION_LOCK = Lock()
_FUSION_CONTEXT = contextlib.ExitStack()
_FUSION_USERS = 0
# Caches napari registered globally for code that computes inside a layer's
# dask_optimized_slicing context. Slices use their own instead.
_REGISTERED_CACHES: set[tuple] = set()
_SLICE_POOL: ThreadPoolExecutor | None = None
_SLICE_POOL_LOCK = Lock()

DaskIndexer = Callable[
    [], contextlib.AbstractContextManager[tuple[dict, Cache | None] | None]
]


@contextlib.contextmanager
def _disable_task_fusion() -> Generator[dict, None, None]:
    global _FUSION_USERS
    # Dask's config dictionary is global. The first slice sets the option,
    # and the last slice restores it, regardless of the order they finish.
    with _FUSION_LOCK:
        if _FUSION_USERS == 0:
            _FUSION_CONTEXT.enter_context(
                dask.config.set({'optimization.fuse.active': False})
            )
        _FUSION_USERS += 1
    try:
        yield dask.config.config
    finally:
        with _FUSION_LOCK:
            _FUSION_USERS -= 1
            if _FUSION_USERS == 0:
                _FUSION_CONTEXT.close()


def _threads_per_slice() -> int:
    """Tasks one slice keeps in flight: Python's default thread pool size.

    Called from a thread other than the main one, Dask creates a pool of this
    size for each calling thread.
    """
    cpus = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
    return min(32, cpus + 4)


def _slice_pool_size() -> int:
    """Threads of the pool shared by all slices on Dask's threaded scheduler."""
    return 2 * _threads_per_slice()


def _slice_scheduler() -> Callable:
    """Dask's local scheduler on the pool shared by all slices.

    Each slice keeps at most as many tasks in flight as Dask's threaded
    scheduler would give it alone. Dask schedules a computation on an
    Executor passed as ``scheduler`` the same way.
    """
    global _SLICE_POOL
    with _SLICE_POOL_LOCK:
        if _SLICE_POOL is None:
            _SLICE_POOL = _PoolExecutor(_slice_pool_size())
            atexit.register(_SLICE_POOL.shutdown)
    return partial(get_async, _SLICE_POOL.submit, _threads_per_slice())


def _dask_chunk_count(data: Any, index: tuple) -> int:
    """Number of Dask chunks that ``data[index]`` reads, or 0 for other data."""
    chunks = getattr(data, 'chunks', None)
    if (
        not dask.is_dask_collection(data)
        or not chunks
        or not isinstance(chunks[0], tuple)
    ):
        return 0
    count = 1
    for dim, sizes in enumerate(chunks):
        key = index[dim] if dim < len(index) else slice(None)
        if not isinstance(key, slice):
            continue  # an integer reads one chunk along this axis
        edges = np.cumsum((0, *sizes))
        start, stop, _ = key.indices(int(edges[-1]))
        if stop <= start:
            return 0
        first = np.searchsorted(edges, start, side='right') - 1
        last = np.searchsorted(edges, stop - 1, side='right') - 1
        count *= int(last - first + 1)
    return count


def _is_local_scheduler(scheduler: Callable) -> bool:
    """Whether ``scheduler`` is Dask's local scheduler, which takes callbacks.

    An Executor configured as Dask's scheduler becomes a partial of it.
    """
    get = scheduler.func if isinstance(scheduler, partial) else scheduler
    return get is get_async or any(
        get is local_scheduler for local_scheduler in named_schedulers.values()
    )


def _slice_callbacks(cache: bool) -> list[tuple]:
    """Callbacks for one slice computation.

    Passed explicitly, they keep the local scheduler from swapping out the
    global ``Callback.active`` while it computes, which affects other
    threads. The user's callbacks stay. napari's cache is scoped to this
    computation instead of registered globally, and left out if the layer
    opts out of caching.
    """
    with _CACHE_LOCK:
        napari_caches = {_DASK_CACHE._callback, *_REGISTERED_CACHES}
    callbacks = [
        callback
        for callback in Callback.active.copy()
        if callback not in napari_caches
    ]
    if cache:
        callbacks.append(_DaskSliceCache(_DASK_CACHE.cache)._callback)
    return callbacks


def _asarray_for_slicing(data: Any, cache: bool | None) -> np.ndarray:
    """Compute ``data`` for a slice.

    ``cache`` is None for data napari doesn't configure for Dask, which is
    computed as ``np.asarray`` would. Otherwise it says whether napari's cache
    keeps the chunks the slice reads.
    """
    if cache is None or not dask.is_dask_collection(data):
        return np.asarray(data)

    # np.asarray on a Dask-backed xarray computes its data array without
    # computing coordinates. Do the same.
    collection = getattr(data, 'data', None)
    if not dask.is_dask_collection(collection):
        collection = data

    # Without task fusion, the cache keeps whole chunks, so the next plane of
    # a chunk doesn't read it again. Dask's config is global, so fusion is
    # off only while this graph is optimized, not while it computes.
    with _disable_task_fusion():
        (collection,) = dask.optimize(collection)

    scheduler = get_scheduler(collections=[collection])
    kwargs = {}
    if _is_local_scheduler(scheduler):
        kwargs['callbacks'] = _slice_callbacks(cache)
    # Otherwise each slicing worker would bring a pool of its own. Slices
    # share the threads, unless the user chose a pool or a number of threads
    # for Dask.
    if (
        scheduler is dask.threaded.get
        and dask.config.get('pool', None) is None
        and dask.config.get('num_workers', None) is None
    ):
        scheduler = _slice_scheduler()
    (result,) = dask.compute(
        collection, scheduler=scheduler, optimize_graph=False, **kwargs
    )
    return np.asarray(result)


def _dask_slice_cache(data: Any, cache: bool) -> bool | None:
    """How slices of ``data`` compute, as ``_asarray_for_slicing`` takes it.

    None if ``data`` isn't Dask data, otherwise ``cache``. Creates napari's
    cache on first use, as ``configure_dask`` does.
    """
    if not _is_dask_data(data):
        return None
    if cache:
        resize_dask_cache()
    return cache


def resize_dask_cache(
    nbytes: int | None = None, mem_fraction: float | None = None
) -> Cache:
    """Create or resize the dask cache used for opportunistic caching.

    The cache object is an instance of a :class:`Cache`, (which
    wraps a :class:`cachey.Cache`).

    See `Dask opportunistic caching
    <https://docs.dask.org/en/latest/caching.html>`_

    Parameters
    ----------
    nbytes : int, optional
        The desired size of the cache, in bytes.  If ``None``, the cache size
        will autodetermined as fraction of the total memory in the system,
        using ``mem_fraction``.  If ``nbytes`` is 0. The cache is turned off.
        by default, cache size is autodetermined using ``mem_fraction``.
    mem_fraction : float, optional
        The fraction (from 0 to 1) of total memory to use for the dask cache.

    Returns
    -------
    dask_cache : dask.cache.Cache
        An instance of a Dask Cache

    Examples
    --------
    >>> from napari.utils import resize_dask_cache
    >>> cache = resize_dask_cache()  # use 25% of total memory by default

    >>> # dask.Cache wraps cachey.Cache
    >>> assert isinstance(cache.cache, cachey.Cache)

    >>> # useful attributes
    >>> cache.cache.available_bytes  # full size of cache
    >>> cache.cache.total_bytes   # currently used bytes
    """
    from psutil import virtual_memory

    if nbytes is None and mem_fraction is not None:
        nbytes = int(virtual_memory().total * mem_fraction)

    with _CACHE_LOCK:
        avail = _DASK_CACHE.cache.available_bytes
        if avail == 1 and nbytes is None:
            nbytes = int(virtual_memory().total * _DEFAULT_MEM_FRACTION)
        if nbytes is not None and nbytes != avail:
            _DASK_CACHE.cache.resize(nbytes)
    return _DASK_CACHE


def _is_dask_data(data: Any) -> bool:
    """Return True if data has a dask computation graph.

    Uses dask's Protocol through `dask.is_dask_collection`, which
    checks for the ``__dask_graph__`` attribute.  This covers things like

    - `dask.array.Array` directly
    - `xarray.DataArray` backed by a dask array (but *not* numpy-backed)
    - Any other dask-collection type
    """
    if dask.is_dask_collection(data):
        return True
    return isinstance(data, collections.abc.Sequence) and any(
        dask.is_dask_collection(i) for i in data
    )


@contextlib.contextmanager
def _registered_cache(dask_cache: Cache) -> Generator[None, None, None]:
    """Register a cache for computations inside ``dask_optimized_slicing``.

    It goes into Dask's global registry, because callers compute with Dask's
    defaults. Slices computed meanwhile leave it out.
    """
    callback = _DaskSliceCache(dask_cache.cache)
    with _CACHE_LOCK:
        _REGISTERED_CACHES.add(callback._callback)
    try:
        with callback:
            yield
    finally:
        with _CACHE_LOCK:
            _REGISTERED_CACHES.discard(callback._callback)


def configure_dask(data: Any, cache: bool = True) -> DaskIndexer:
    """Spin up cache and return context manager that optimizes Dask indexing.

    This function determines whether data is a dask array or list of dask
    arrays and prepares some optimizations if so.

    When a delayed dask array is given to napari, there are couple things that
    need to be done to optimize performance.

    1. Opportunistic caching needs to be enabled, such that we don't recompute
       (or "re-read") data that has already been computed or read.

    2. Dask task fusion must be turned off to prevent napari from triggering
       new io on data that has already been read from disk. For example, with a
       4D timelapse of 3D stacks, napari may actually *re-read* the entire 3D
       tiff file every time the Z plane index is changed. Turning of Dask task
       fusion with ``optimization.fuse.active == False`` prevents this.

       .. note::

          Turning off task fusion requires Dask version 2.15.0 or later.

    For background and context, see `napari/napari#718
    <https://github.com/napari/napari/issues/718>`_, `napari/napari#1124
    <https://github.com/napari/napari/pull/1124>`_, and `dask/dask#6084
    <https://github.com/dask/dask/pull/6084>`_.

    For details on Dask task fusion, see the documentation on `Dask
    Optimization <https://docs.dask.org/en/latest/optimize.html>`_.

    Parameters
    ----------
    data : Any
        data, as passed to a ``Layer.__init__`` method.
    cache: bool
        Whether to use the global dask cache.  If False, no cache will be used

    Returns
    -------
    ContextManager
        A context manager that can be used to optimize dask indexing

    Examples
    --------
    >>> data = dask.array.ones((10,10,10))
    >>> optimized_slicing = configure_dask(data)
    >>> with optimized_slicing():
    ...    data[0, 2].compute()
    """
    if not _is_dask_data(data):
        return contextlib.nullcontext

    dask_cache = resize_dask_cache() if cache else None

    @contextlib.contextmanager
    def dask_optimized_slicing(
        memfrac: float = 0.5,
    ) -> Generator[tuple[dict, Cache | None], None, None]:
        with _disable_task_fusion() as cfg:
            if dask_cache is None:
                yield cfg, None
            else:
                with _registered_cache(dask_cache):
                    yield cfg, dask_cache

    return dask_optimized_slicing
