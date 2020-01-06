"""Miscellaneous utility functions.
"""
from enum import Enum, EnumMeta
import re
import inspect
import itertools
import numpy as np
from copy import deepcopy

try:
    # SIGNIFICANTLY faster array hashing, but requires a new dependency
    import xxhash
except ImportError:
    xxhash = None


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
            value = value.lower()
            return super().__call__(value)
        # otherwise create new Enum class
        return cls._create_(
            value,
            names,
            module=module,
            qualname=qualname,
            type=type,
            start=start,
        )


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


# https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array
def hash_array(obj, complete=False):
    """Hash a numpy array.

    Parameters
    ----------
    obj : np.ndarray
        the array to hash
    complete : bool, optional
        whether to take the whole array into account, or use str(obj), as a
        shortcut. by default False

    Returns
    -------
    int
        the array hash
    """
    # XXX: This is a potential gotcha.  If we want to include data arrays in
    # hashes (which would make it easier to know when to update, for instance,
    # a remote viewer), then it can begin to take a long time (many seconds) to
    # hash even moderately-sized arrays.  xxhash is very fast, but requires a
    # new dependency.  However, if we *don't* ship with xxhash, then hashes of
    # identical arrays will be different on two systems if one has xxhash.
    # we should probably be either all-in or all-out.
    if xxhash:
        h = xxhash.xxh64()
        h.update(obj)
        return h.intdigest()
    if complete:
        # take whole array into account.  Can be slow on large arrays
        return hash(obj.data.tobytes())
    # otherwise hash just the string (does not take full array into account)
    return hash(str(obj))


# https://stackoverflow.com/a/8714242/1631624
def recursive_hash(obj, skip_arrays=False):
    """Recursively hash a (possibly nested) dictionary, list, tuple or set.

    Only works if all (nested) values in object are also hashable types
    (lists, tuples, sets, dictionaries, or np.arrays).

    Parameters
    ----------
    obj : object
        The object to hash

    Returns
    -------
    int
        the hash
    """
    if isinstance(obj, (set, tuple, list)):
        return tuple([recursive_hash(e, skip_arrays=skip_arrays) for e in obj])
    elif isinstance(obj, np.ndarray):
        if skip_arrays:
            return 0
        else:
            return hash_array(obj)
    elif isinstance(obj, dict):
        new_o = deepcopy(obj)
        for k, v in new_o.items():
            new_o[k] = recursive_hash(v, skip_arrays=skip_arrays)
        return hash(tuple(frozenset(sorted(new_o.items()))))
    else:
        return hash(obj)
