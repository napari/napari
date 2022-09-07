"""Miscellaneous utility functions.
"""

import builtins
import collections.abc
import contextlib
import importlib.metadata
import inspect
import itertools
import os
import re
import sys
import warnings
from enum import Enum, EnumMeta
from os import fspath
from os import path as os_path
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

import numpy as np

from ..utils.translations import trans

if TYPE_CHECKING:
    import packaging.version


ROOT_DIR = os_path.dirname(os_path.dirname(__file__))


def parse_version(v) -> 'packaging.version._BaseVersion':
    """Parse a version string and return a packaging.version.Version obj."""
    import packaging.version

    try:
        return packaging.version.Version(v)
    except packaging.version.InvalidVersion:
        return packaging.version.LegacyVersion(v)


def running_as_bundled_app() -> bool:
    """Infer whether we are running as a briefcase bundle."""
    # https://github.com/beeware/briefcase/issues/412
    # https://github.com/beeware/briefcase/pull/425
    # note that a module may not have a __package__ attribute
    # From 0.4.12 we add a sentinel file next to the bundled sys.executable
    if (Path(sys.executable).parent / ".napari_is_bundled").exists():
        return True

    try:
        app_module = sys.modules['__main__'].__package__
    except AttributeError:
        return False

    try:
        metadata = importlib.metadata.metadata(app_module)
    except importlib.metadata.PackageNotFoundError:
        return False

    return 'Briefcase-Version' in metadata


def running_as_constructor_app() -> bool:
    """Infer whether we are running as a constructor bundle."""
    return (
        Path(sys.prefix).parent.parent / ".napari_is_bundled_constructor"
    ).exists()


def bundle_bin_dir() -> Optional[str]:
    """Return path to briefcase app_packages/bin if it exists."""
    bin = os_path.join(os_path.dirname(sys.exec_prefix), 'app_packages', 'bin')
    if os_path.isdir(bin):
        return bin


def in_jupyter() -> bool:
    """Return true if we're running in jupyter notebook/lab or qtconsole."""
    try:
        from IPython import get_ipython

        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except Exception:
        pass
    return False


def in_ipython() -> bool:
    """Return true if we're running in an IPython interactive shell."""
    try:
        from IPython import get_ipython

        return get_ipython().__class__.__name__ == 'TerminalInteractiveShell'
    except Exception:
        pass
    return False


def str_to_rgb(arg):
    """Convert an rgb string 'rgb(x,y,z)' to a list of ints [x,y,z]."""
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


def is_iterable(arg, color=False, allow_none=False):
    """Determine if a single argument is an iterable. If a color is being
    provided and the argument is a 1-D array of length 3 or 4 then the input
    is taken to not be iterable. If allow_none is True, `None` is considered iterable.
    """
    if arg is None and not allow_none:
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


def ensure_sequence_of_iterables(
    obj,
    length: Optional[int] = None,
    repeat_empty: bool = False,
    allow_none: bool = False,
):
    """Ensure that ``obj`` behaves like a (nested) sequence of iterables.

    If length is provided and the object is already a sequence of iterables,
    a ValueError will be raised if ``len(obj) != length``.

    Parameters
    ----------
    obj : Any
        the object to check
    length : int, optional
        If provided, assert that obj has len ``length``, by default None
    repeat_empty : bool
        whether to repeat an empty sequence (otherwise return the empty sequence itself)
    allow_none : bool
        treat None as iterable

    Returns
    -------
    iterable
        nested sequence of iterables, or an itertools.repeat instance

    Examples
    --------
    In [1]: ensure_sequence_of_iterables([1, 2])
    Out[1]: repeat([1, 2])

    In [2]: ensure_sequence_of_iterables([(1, 2), (3, 4)])
    Out[2]: [(1, 2), (3, 4)]

    In [3]: ensure_sequence_of_iterables([(1, 2), None], allow_none=True)
    Out[3]: [(1, 2), None]

    In [4]: ensure_sequence_of_iterables({'a':1})
    Out[4]: repeat({'a': 1})

    In [5]: ensure_sequence_of_iterables(None)
    Out[5]: repeat(None)

    In [6]: ensure_sequence_of_iterables([])
    Out[6]: repeat([])

    In [7]: ensure_sequence_of_iterables([], repeat_empty=False)
    Out[7]: []
    """

    if (
        obj is not None
        and is_sequence(obj)
        and all(is_iterable(el, allow_none=allow_none) for el in obj)
    ):
        if length is not None and len(obj) != length:
            if (len(obj) == 0 and not repeat_empty) or len(obj) > 0:
                # sequence of iterables of wrong length
                raise ValueError(
                    trans._(
                        "length of {obj} must equal {length}",
                        deferred=True,
                        obj=obj,
                        length=length,
                    )
                )

        if len(obj) > 0 or not repeat_empty:
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
        """set the item name case to uppercase for name lookup"""
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
        """set the item value case to lowercase for value lookup"""
        # simple value lookup
        if names is None:
            if isinstance(value, str):
                return super().__call__(value.lower())
            elif isinstance(value, cls):
                return value
            else:
                raise ValueError(
                    trans._(
                        '{class_name} may only be called with a `str` or an instance of {class_name}. Got {dtype}',
                        deferred=True,
                        class_name=cls,
                        dtype=builtins.type(value),
                    )
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
        """autonaming function assigns each value its own name as a value"""
        return name.lower()

    def __str__(self):
        """String representation: The string method returns the lowercase
        string of the Enum name
        """
        return self.value

    def __eq__(self, other):
        if type(self) is type(other):
            return self is other
        elif isinstance(other, str):
            return str(self) == other
        return NotImplemented

    def __hash__(self):
        return hash(str(self))


camel_to_snake_pattern = re.compile(r'(.)([A-Z][a-z]+)')
camel_to_spaces_pattern = re.compile(
    r"((?<=[a-z])[A-Z]|(?<!\A)[A-R,T-Z](?=[a-z]))"
)


def camel_to_snake(name):
    # https://gist.github.com/jaytaylor/3660565
    return camel_to_snake_pattern.sub(r'\1_\2', name).lower()


def camel_to_spaces(val):
    return camel_to_spaces_pattern.sub(r" \1", val)


T = TypeVar('T', str, Path)


def abspath_or_url(relpath: T, *, must_exist: bool = False) -> T:
    """Utility function that normalizes paths or a sequence thereof.

    Expands user directory and converts relpaths to abspaths... but ignores
    URLS that begin with "http", "ftp", or "file".

    Parameters
    ----------
    relpath : str|Path
        A path, either as string or Path object.
    must_exist : bool, default True
        Raise ValueError if `relpath` is not a URL and does not exist.

    Returns
    -------
    abspath : str|Path
        An absolute path, or list or tuple of absolute paths (same type as
        input)
    """
    from urllib.parse import urlparse

    if not isinstance(relpath, (str, Path)):
        raise TypeError(
            trans._("Argument must be a string or Path", deferred=True)
        )
    OriginType = type(relpath)

    relpath = fspath(relpath)
    urlp = urlparse(relpath)
    if urlp.scheme and urlp.netloc:
        return relpath

    path = os_path.abspath(os_path.expanduser(relpath))
    if must_exist and not (urlp.scheme or urlp.netloc or os.path.exists(path)):
        raise ValueError(
            trans._(
                "Requested path {path!r} does not exist.",
                deferred=True,
                path=path,
            )
        )
    return OriginType(path)


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
            formatted = f'{formatted}={formatted}'

        if kind == inspect._VAR_POSITIONAL:
            formatted = '*' + formatted
        elif kind == inspect._VAR_KEYWORD:
            formatted = '**' + formatted

        return formatted


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


def ensure_n_tuple(val, n, fill=0):
    """Ensure input is a length n tuple.

    Parameters
    ----------
    val : iterable
        Iterable to be forced into length n-tuple.
    n : int
        Length of tuple.

    Returns
    -------
    tuple
        Coerced tuple.
    """
    assert n > 0, 'n must be greater than 0'
    tuple_value = tuple(val)
    return (fill,) * (n - len(tuple_value)) + tuple_value[-n:]


def ensure_layer_data_tuple(val):
    msg = trans._(
        'Not a valid layer data tuple: {value!r}',
        deferred=True,
        value=val,
    )
    if not isinstance(val, tuple) and val:
        raise TypeError(msg)
    if len(val) > 1:
        if not isinstance(val[1], dict):
            raise TypeError(msg)
        if len(val) > 2 and not isinstance(val[2], str):
            raise TypeError(msg)
    return val


def ensure_list_of_layer_data_tuple(val) -> List[tuple]:
    # allow empty list to be returned but do nothing in that case
    if isinstance(val, list):
        with contextlib.suppress(TypeError):
            return [ensure_layer_data_tuple(v) for v in val]
    raise TypeError(
        trans._('Not a valid list of layer data tuples!', deferred=True)
    )


def _quiet_array_equal(*a, **k):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "elementwise comparison")
        return np.array_equal(*a, **k)


def _arraylike_short_names(obj) -> Iterator[str]:
    """Yield all the short names of an array-like or its class."""
    type_ = type(obj) if not inspect.isclass(obj) else obj
    for base in type_.mro():
        yield f'{base.__module__.split(".", maxsplit=1)[0]}.{base.__name__}'


def pick_equality_operator(obj) -> Callable[[Any, Any], bool]:
    """Return a function that can check equality between ``obj`` and another.

    Rather than always using ``==`` (i.e. ``operator.eq``), this function
    returns operators that are aware of object types: mostly "array types with
    more than one element" whose truth value is ambiguous.

    This function works for both classes (types) and instances.  If an instance
    is passed, it will be first cast to a type with type(obj).

    Parameters
    ----------
    obj : Any
        An object whose equality with another object you want to check.

    Returns
    -------
    operator : Callable[[Any, Any], bool]
        An operation that can be called as ``operator(obj, other)`` to check
        equality between objects of type ``type(obj)``.
    """
    import operator

    # yes, it's a little riskier, but we are checking namespaces instead of
    # actual `issubclass` here to avoid slow import times
    _known_arrays = {
        'numpy.ndarray': _quiet_array_equal,  # numpy.ndarray
        'dask.Array': operator.is_,  # dask.array.core.Array
        'dask.Delayed': operator.is_,  # dask.delayed.Delayed
        'zarr.Array': operator.is_,  # zarr.core.Array
        'xarray.DataArray': _quiet_array_equal,  # xarray.core.dataarray.DataArray
    }

    for name in _arraylike_short_names(obj):
        func = _known_arrays.get(name)
        if func:
            return func

    return operator.eq


def _is_array_type(array, type_name: str) -> bool:
    """Checks if an array-like instance or class is of the type described by a short name.

    This is useful when you want to check the type of array-like quickly without
    importing its package, which might take a long time.

    Parameters
    ----------
    array
        The array-like object.
    type_name : str
        The short name of the type to test against (e.g. 'numpy.ndarray', 'xarray.DataArray').

    Returns
    -------
    True if the array is associated with the type name.
    """
    return type_name in _arraylike_short_names(array)


def dir_hash(
    path: Union[str, Path], include_paths=True, ignore_hidden=True
) -> str:
    """Compute the hash of a directory, based on structure and contents.

    Parameters
    ----------
    path : Union[str, Path]
        Source path which will be used to select all files (and files in subdirectories)
        to compute the hexadecimal digest.
    include_paths : bool
        If ``True``, the hash will also include the ``file`` parts.
    ignore_hidden : bool
        If ``True``, hidden files (starting with ``.``) will be ignored when
        computing the hash.

    Returns
    -------
    hash : str
        Hexadecimal digest of all files in the provided path.
    """
    import hashlib

    if not Path(path).is_dir():
        raise TypeError(
            trans._(
                "{path} is not a directory.",
                deferred=True,
                path=path,
            )
        )

    hash_func = hashlib.md5
    _hash = hash_func()
    for root, _, files in os.walk(path):
        for fname in sorted(files):
            if fname.startswith(".") and ignore_hidden:
                continue
            _file_hash(_hash, Path(root) / fname, path, include_paths)
    return _hash.hexdigest()


def paths_hash(
    paths: Iterable[Union[str, Path]],
    include_paths: bool = True,
    ignore_hidden: bool = True,
) -> str:
    """Compute the hash of list of paths.

    Parameters
    ----------
    paths : Iterable[Union[str, Path]]
        An iterable of paths to files which will be used when computing the hash.
    include_paths : bool
        If ``True``, the hash will also include the ``file`` parts.
    ignore_hidden : bool
        If ``True``, hidden files (starting with ``.``) will be ignored when
        computing the hash.

    Returns
    -------
    hash : str
        Hexadecimal digest of the contents of provided files.
    """
    import hashlib

    hash_func = hashlib.md5
    _hash = hash_func()
    for file_path in sorted(paths):
        file_path = Path(file_path)
        if ignore_hidden and str(file_path.stem).startswith("."):
            continue
        _file_hash(_hash, file_path, file_path.parent, include_paths)
    return _hash.hexdigest()


def _file_hash(_hash, file: Path, path: Path, include_paths: bool = True):
    """Update hash with based on file contents and optionally relative path.

    Parameters
    ----------
    _hash
    file : Path
        Path to the source file which will be used to compute the hash.
    path : Path
        Path to the base directory of the `file`. This can be usually obtained by using `file.parent`.
    include_paths : bool
        If ``True``, the hash will also include the ``file`` parts.
    """
    _hash.update(file.read_bytes())

    if include_paths:
        # update the hash with the filename
        fparts = file.relative_to(path).parts
        _hash.update(''.join(fparts).encode())


def _combine_signatures(
    *objects: Callable, return_annotation=inspect.Signature.empty, exclude=()
) -> inspect.Signature:
    """Create combined Signature from objects, excluding names in `exclude`.

    Parameters
    ----------
    *objects : Callable
        callables whose signatures should be combined
    return_annotation : [type], optional
        The return annotation to use for combined signature, by default
        inspect.Signature.empty (as it's ambiguous)
    exclude : tuple, optional
        Parameter names to exclude from the combined signature (such as
        'self'), by default ()

    Returns
    -------
    inspect.Signature
        Signature object with the combined signature. Reminder, str(signature)
        provides a very nice repr for code generation.
    """
    params = itertools.chain(
        *(inspect.signature(o).parameters.values() for o in objects)
    )
    new_params = sorted(
        (p for p in params if p.name not in exclude),
        key=lambda p: p.kind,
    )
    return inspect.Signature(new_params, return_annotation=return_annotation)


def deep_update(dct: dict, merge_dct: dict, copy=True) -> dict:
    """Merge possibly nested dicts"""
    _dct = dct.copy() if copy else dct
    for k, v in merge_dct.items():
        if k in _dct and isinstance(dct[k], dict) and isinstance(v, dict):
            deep_update(_dct[k], v, copy=False)
        else:
            _dct[k] = v
    return _dct


def install_certifi_opener():
    """Install urlopener that uses certifi context.

    This is useful in the bundle, where otherwise users might get SSL errors
    when using `urllib.request.urlopen`.
    """
    import ssl
    from urllib import request

    import certifi

    context = ssl.create_default_context(cafile=certifi.where())
    https_handler = request.HTTPSHandler(context=context)
    opener = request.build_opener(https_handler)
    request.install_opener(opener)
