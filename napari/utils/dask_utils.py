"""Dask cache utilities.
"""
import warnings
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Callable, ContextManager, Optional

import dask
import dask.array as da
from dask.cache import Cache

from .. import utils
from ..utils.translations import trans


def create_dask_cache(
    nbytes: Optional[int] = None, mem_fraction: float = 0.1
) -> Cache:
    """Create a dask cache at utils.dask_cache if one doesn't already exist.

    Parameters
    ----------
    nbytes : int, optional
        The desired size of the cache, in bytes.  If ``None``, the cache size
        will autodetermined as fraction of the total memory in the system,
        using ``mem_fraction``.  If ``nbytes`` is 0, cache object will be
        created, but not caching will occur. by default, cache size is
        autodetermined using ``mem_fraction``.
    mem_fraction : float, optional
        The fraction (from 0 to 1) of total memory to use for the dask cache.
        by default, 10% of total memory is used.

    Returns
    -------
    dask_cache : dask.cache.Cache
        An instance of a Dask Cache
    """
    import psutil

    if nbytes is None:
        nbytes = psutil.virtual_memory().total * mem_fraction
    if not (
        hasattr(utils, 'dask_cache') and isinstance(utils.dask_cache, Cache)
    ):
        utils.dask_cache = Cache(nbytes)
        utils.dask_cache.register()
    return utils.dask_cache


def resize_dask_cache(
    nbytes: Optional[int] = None, mem_fraction: float = None
) -> Cache:
    """Create or resize the dask cache used for opportunistic caching.

    The cache object is an instance of a :class:`Cache`, (which
    wraps a :class:`cachey.Cache`), and is made available at
    :attr:`napari.utils.dask_cache`.

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
    >>> cache = resize_dask_cache()  # use 50% of total memory by default

    >>> # dask.Cache wraps cachey.Cache
    >>> assert isinstance(cache.cache, cachey.Cache)

    >>> # useful attributes
    >>> cache.cache.available_bytes  # full size of cache
    >>> cache.cache.total_bytes   # currently used bytes
    """

    import psutil

    if nbytes is None and mem_fraction is not None:
        nbytes = psutil.virtual_memory().total * mem_fraction

    # if we don't have a cache already, create one.  If neither nbytes nor
    # mem_fraction was provided, it will use the default size as determined in
    # create_cache.
    if not (
        hasattr(utils, 'dask_cache') and isinstance(utils.dask_cache, Cache)
    ):
        return create_dask_cache(nbytes)
    else:  # we already have a cache
        # if the cache has already been registered, then calling
        # resize_dask_cache() without supplying either mem_fraction or nbytes
        # is a no-op:
        if (
            nbytes is not None
            and nbytes != utils.dask_cache.cache.available_bytes
        ):
            utils.dask_cache.cache.resize(nbytes)

    return utils.dask_cache


def _is_dask_data(data) -> bool:
    """Return True if data is a dask array or a list/tuple of dask arrays."""
    return isinstance(data, da.Array) or (
        isinstance(data, (list, tuple))
        and any(isinstance(i, da.Array) for i in data)
    )


def configure_dask(data) -> Callable[[], ContextManager[dict]]:
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
    if _is_dask_data(data):
        if dask.__version__ < LooseVersion('2.15.0'):
            warnings.warn(
                trans._(
                    'For best performance with Dask arrays in napari, please upgrade Dask to v2.15.0 or later. Current version is {dask_version}',
                    deferred=True,
                    dask_version=dask.__version__,
                )
            )

        def dask_optimized_slicing():
            with dask.config.set({"optimization.fuse.active": False}) as cfg:
                yield cfg

    else:

        def dask_optimized_slicing():
            yield {}

    return contextmanager(dask_optimized_slicing)
