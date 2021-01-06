"""Miscellaneous utility functions.
"""
import builtins
import collections.abc
import inspect
import itertools
import re
import sys
from enum import Enum, EnumMeta
from os import PathLike, fspath, path
from typing import Optional, Sequence, Type, TypeVar
from urllib.parse import urlparse

import numpy as np

ROOT_DIR = path.dirname(path.dirname(__file__))

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import importlib_metadata  # noqa


def running_as_bundled_app() -> bool:
    """Infer whether we are running as a briefcase bundle"""
    # https://github.com/beeware/briefcase/issues/412
    # https://github.com/beeware/briefcase/pull/425
    app_module = sys.modules['__main__'].__package__
    try:
        metadata = importlib_metadata.metadata(app_module)
    except importlib_metadata.PackageNotFoundError:
        return False

    return 'Briefcase-Version' in metadata


def bundle_bin_dir() -> Optional[str]:
    """Return path to briefcase app_packages/bin if it exists."""
    bin = path.join(path.dirname(sys.exec_prefix), 'app_packages', 'bin')
    if path.isdir(bin):
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
    """

    if obj is not None and is_sequence(obj) and is_iterable(obj[0]):
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
                    f'{cls} may only be called with a `str`'
                    f' or an instance of {cls}. Got {builtins.type(value)}'
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


camel_to_snake_pattern = re.compile(r'(.)([A-Z][a-z]+)')
camel_to_spaces_pattern = re.compile(
    r"((?<=[a-z])[A-Z]|(?<!\A)[A-R,T-Z](?=[a-z]))"
)


def camel_to_snake(name):
    # https://gist.github.com/jaytaylor/3660565
    return camel_to_snake_pattern.sub(r'\1_\2', name).lower()


def camel_to_spaces(val):
    return camel_to_spaces_pattern.sub(r" \1", val)


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
        urlp = urlparse(relpath)
        if urlp.scheme and urlp.netloc:
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
            formatted = f'{formatted}={formatted}'

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
            rendered += f' -> {anno}'

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
    if not (isinstance(val, tuple) and (0 < len(val) <= 3)):
        raise TypeError(f'Not a valid layer data tuple: {val!r}')
    return val


def ensure_list_of_layer_data_tuple(val):
    if isinstance(val, list) and len(val):
        try:
            return [ensure_layer_data_tuple(v) for v in val]
        except TypeError:
            pass
    raise TypeError('Not a valid list of layer data tuples!')
