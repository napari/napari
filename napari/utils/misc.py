"""Miscellaneous utility functions.
"""
import collections.abc
import inspect
import itertools
import re
import warnings
from contextlib import contextmanager
from enum import Enum, EnumMeta
from os import PathLike, fspath, path
from typing import ContextManager, Optional, Sequence, Type, TypeVar

import dask
import dask.array as da
import numpy as np
from dask.cache import Cache

from .. import utils

ROOT_DIR = path.dirname(path.dirname(__file__))


def str_to_rgb(arg):
    """Convert an rgb string 'rgb(x,y,z)' to a list of ints [x,y,z].
    """
    return list(
        map(int, re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', arg).groups())
    )


def ensure_iterable(arg, color=False):
    """Ensure an argument is an iterable. Useful when an input argument
    can either be a single value or a list. If a color is passed then it
    will be treated specially to determine if it is iterable.
    """
    if is_iterable(arg, color=color):
        return arg
    else:
        return itertools.repeat(arg)


def is_iterable(arg, color=False):
    """Determine if a single argument is an iterable. If a color is being
    provided and the argument is a 1-D array of length 3 or 4 then the input
    is taken to not be iterable.
    """
    if arg is None:
        return False
    elif type(arg) is str:
        return False
    elif np.isscalar(arg):
        return False
    elif color and isinstance(arg, (list, np.ndarray)):
        if np.array(arg).ndim == 1 and (len(arg) == 3 or len(arg) == 4):
            return False
        else:
            return True
    else:
        return True


def is_sequence(arg):
    """Check if ``arg`` is a sequence like a list or tuple.

    return True:
        list
        tuple
    return False
        string
        numbers
        dict
        set
    """
    if isinstance(arg, collections.abc.Sequence) and not isinstance(arg, str):
        return True
    return False


def ensure_sequence_of_iterables(obj, length: Optional[int] = None):
    """Ensure that ``obj`` behaves like a (nested) sequence of iterables.

    If length is provided and the object is already a sequence of iterables,
    a ValueError will be raised if ``len(obj) != length``.

    Examples
    --------
    In [1]: ensure_sequence_of_iterables([1, 2])
    Out[1]: repeat([1, 2])

    In [2]: ensure_sequence_of_iterables([(1, 2), (3, 4)])
    Out[2]: [(1, 2), (3, 4)]

    In [3]: ensure_sequence_of_iterables({'a':1})
    Out[3]: repeat({'a': 1})

    In [4]: ensure_sequence_of_iterables(None)
    Out[4]: repeat(None)

    Parameters
    ----------
    obj : Any
        the object to check
    length : int, optional
        If provided, assert that obj has len ``length``, by default None

    Returns
    -------
    iterable
        nested sequence of iterables, or an itertools.repeat instance
    """
    if obj and is_sequence(obj) and is_iterable(obj[0]):
        if length is not None and len(obj) != length:
            raise ValueError(f"length of {obj} must equal {length}")
        return obj
    return itertools.repeat(obj)


def formatdoc(obj):
    """Substitute globals and locals into an object's docstring."""
    frame = inspect.currentframe().f_back
    try:
        obj.__doc__ = obj.__doc__.format(
            **{**frame.f_globals, **frame.f_locals}
        )
        return obj
    finally:
        del frame


class StringEnumMeta(EnumMeta):
    def __getitem__(self, item):
        """ set the item name case to uppercase for name lookup
        """
        if isinstance(item, str):
            item = item.upper()

        return super().__getitem__(item)

    def __call__(
        cls,
        value,
        names=None,
        *,
        module=None,
        qualname=None,
        type=None,
        start=1,
    ):
        """ set the item value case to lowercase for value lookup
        """
        # simple value lookup
        if names is None:
            if isinstance(value, str):
                return super().__call__(value.lower())
            elif isinstance(value, cls):
                return value
            else:
                raise ValueError(
                    f'{cls} may only be called with a `str`'
                    f' or an instance of {cls}'
                )

        # otherwise create new Enum class
        return cls._create_(
            value,
            names,
            module=module,
            qualname=qualname,
            type=type,
            start=start,
        )

    def keys(self):
        return list(map(str, self))


class StringEnum(Enum, metaclass=StringEnumMeta):
    def _generate_next_value_(name, start, count, last_values):
        """ autonaming function assigns each value its own name as a value
        """
        return name.lower()

    def __str__(self):
        """String representation: The string method returns the lowercase
        string of the Enum name
        """
        return self.value


camel_to_snake_pattern = re.compile(r'(.)([A-Z][a-z]+)')


def camel_to_snake(name):
    # https://gist.github.com/jaytaylor/3660565
    return camel_to_snake_pattern.sub(r'\1_\2', name).lower()


T = TypeVar('T', str, Sequence[str])


def abspath_or_url(relpath: T) -> T:
    """Utility function that normalizes paths or a sequence thereof.

    Expands user directory and converts relpaths to abspaths... but ignores
    URLS that begin with "http", "ftp", or "file".

    Parameters
    ----------
    relpath : str or list or tuple
        A path, or list or tuple of paths.

    Returns
    -------
    abspath : str or list or tuple
        An absolute path, or list or tuple of absolute paths (same type as
        input).
    """
    if isinstance(relpath, (tuple, list)):
        return type(relpath)(abspath_or_url(p) for p in relpath)

    if isinstance(relpath, (str, PathLike)):
        relpath = fspath(relpath)
        if relpath.startswith(('http:', 'https:', 'ftp:', 'file:')):
            return relpath
        return path.abspath(path.expanduser(relpath))

    raise TypeError("Argument must be a string, PathLike, or sequence thereof")


class CallDefault(inspect.Parameter):
    def __str__(self):
        """wrap defaults"""
        kind = self.kind
        formatted = self._name

        # Fill in defaults
        if (
            self._default is not inspect._empty
            or kind == inspect._KEYWORD_ONLY
        ):
            formatted = '{}={}'.format(formatted, formatted)

        if kind == inspect._VAR_POSITIONAL:
            formatted = '*' + formatted
        elif kind == inspect._VAR_KEYWORD:
            formatted = '**' + formatted

        return formatted


class CallSignature(inspect.Signature):
    _parameter_cls = CallDefault

    def __str__(self):
        """do not render separators

        commented code is what was taken out from
        the copy/pasted inspect module code :)
        """
        result = []
        # render_pos_only_separator = False
        # render_kw_only_separator = True
        for param in self.parameters.values():
            formatted = str(param)
            result.append(formatted)

        rendered = '({})'.format(', '.join(result))

        if self.return_annotation is not inspect._empty:
            anno = inspect.formatannotation(self.return_annotation)
            rendered += ' -> {}'.format(anno)

        return rendered


callsignature = CallSignature.from_callable


def all_subclasses(cls: Type) -> set:
    """Recursively find all subclasses of class ``cls``.

    Parameters
    ----------
    cls : class
        A python class (or anything that implements a __subclasses__ method).

    Returns
    -------
    set
        the set of all classes that are subclassed from ``cls``
    """
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )


def create_dask_cache(nbytes=None, mem_fraction=0.5):
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
        by default, 50% of total memory is used.

    Returns
    -------
    [type]
        [description]
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

    See `Dask oportunistic caching
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
    dask_cache : Cache
        An instance of a Dask Cache

    Example
    -------
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


def configure_dask(data) -> ContextManager[dict]:
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

    Example
    -------
    >>> data = dask.array.ones((10,10,10))
    >>> optimized_slicing = configure_dask(data)
    >>> with optimized_slicing():
    ...    data[0, 2].compute()
    """
    if _is_dask_data(data):
        create_dask_cache()  # creates one if it doesn't exist
        dask_version = tuple(map(int, dask.__version__.split(".")))
        if dask_version < (2, 15, 0):
            warnings.warn(
                'For best performance with Dask arrays in napari, please '
                'upgrade Dask to v2.15.0 or later. Current version is '
                f'{dask.__version__}'
            )

        def dask_optimized_slicing(*args, **kwds):
            with dask.config.set({"optimization.fuse.active": False}) as cfg:
                yield cfg

    else:

        def dask_optimized_slicing(*args, **kwds):
            yield {}

    return contextmanager(dask_optimized_slicing)
