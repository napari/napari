import builtins
import logging
import os
from collections import abc
from typing import Iterable, Optional, TypeVar

logger = logging.getLogger('napari.config')


def canonical_name(k: str, config: abc.Mapping) -> str:
    """Return the canonical name for a key.

    Handles user choice of '-' or '_' conventions by standardizing on whichever
    version was set first. If a key already exists in either hyphen or
    underscore form, the existing version is the canonical name. If neither
    version exists the original key is used as is.
    """
    try:
        if k in config:
            return k
    except TypeError:
        # config is not a mapping, return the same name as provided
        return k

    altk = k.replace("_", "-") if "_" in k else k.replace("-", "_")

    if altk in config:
        return altk

    return k


def update(old: dict, new: dict, priority="new") -> dict:
    """Update a nested dictionary with values from another.

    This is like dict.update except that it smoothly merges nested values

    This operates in-place and modifies old

    Parameters
    ----------
    priority: string {'old', 'new'}
        If new (default) then the new dictionary has preference.
        Otherwise the old dictionary does.

    Examples
    --------
    >>> a = {'x': 1, 'y': {'a': 2}}
    >>> b = {'x': 2, 'y': {'b': 3}}
    >>> update(a, b)
    {'x': 2, 'y': {'a': 2, 'b': 3}}

    >>> a = {'x': 1, 'y': {'a': 2}}
    >>> b = {'x': 2, 'y': {'b': 3}}
    >>> update(a, b, priority='old')
    {'x': 1, 'y': {'a': 2, 'b': 3}}

    See Also
    --------
    napari.config.merge
    """
    for k, v in new.items():
        k = canonical_name(k, old)

        if isinstance(v, abc.Mapping):
            if k not in old or old[k] is None:
                old[k] = {}
            update(old[k], v, priority=priority)  # type: ignore
        else:
            if priority == "new" or k not in old:
                old[k] = v

    return old


def merge(*dicts: dict) -> dict:
    """Update a sequence of nested dictionaries.

    This prefers the values in the latter dictionaries to those in the former

    Examples
    --------
    >>> a = {'x': 1, 'y': {'a': 2}}
    >>> b = {'y': {'b': 3}}
    >>> merge(a, b)
    {'x': 1, 'y': {'a': 2, 'b': 3}}

    See Also
    --------
    napari.config.update
    """
    result: dict = {}
    for d in dicts:
        update(result, d)
    return result


def ensure_file(source: str, destination: Optional[str] = None, comment=True):
    """Copy file to default location if it does not already exist.

    This tries to move a default configuration file to a default location if if
    does not already exist.  It also comments out that file by default.

    This is to be used by downstream modules that may have default
    configuration files that they wish to include in the default configuration
    path.

    Parameters
    ----------
    source : string, filename
        Source configuration file, typically within a source directory.
    destination : string, directory
        Destination directory. Configurable by ``NAPARI_CONFIG`` environment
        variable, falling back to ~/.config/napari.
    comment : bool, True by default
        Whether or not to comment out the config file when copying.
    """
    if destination is None:
        from ..config import core

        destination = core.PATH

    # destination is a file and already exists, never overwrite
    if os.path.isfile(destination):
        return

    # If destination is not an existing file, interpret as a directory,
    # use the source basename as the filename
    directory = destination
    destination = os.path.join(directory, os.path.basename(source))

    try:
        if not os.path.exists(destination):
            logger.debug('copying %r to %r' % (source, destination))
            os.makedirs(directory, exist_ok=True)

            # Atomically create destination.  Parallel testing discovered
            # a race condition where a process can be busy creating the
            # destination while another process reads an empty config file.
            tmp = "%s.tmp.%d" % (destination, os.getpid())
            with open(source) as f:
                lines = list(f)

            if comment:
                lines = [
                    "# " + line
                    if line.strip() and not line.startswith("#")
                    else line
                    for line in lines
                ]

            with open(tmp, "w") as f:
                f.write("".join(lines))

            try:
                os.rename(tmp, destination)
            except OSError:
                os.remove(tmp)
    except (IOError, OSError):
        pass


T = TypeVar("T", abc.Mapping, Iterable, str)


def expand_environment_variables(config: T) -> T:
    """Expand environment variables in a nested config dictionary.

    This function will recursively search through any nested dictionaries
    and/or lists.

    Parameters
    ----------
    config : dict, iterable, or str
        Input object to search for environment variables

    Returns
    -------
    config : same type as input

    Examples
    --------
    >>> expand_environment_variables({'x': [1, 2, '$USER']})
    {'x': [1, 2, 'my-username']}
    """
    if isinstance(config, abc.Mapping):
        return {k: expand_environment_variables(v) for k, v in config.items()}
    elif isinstance(config, str):
        return os.path.expandvars(config)
    elif isinstance(config, (list, tuple, builtins.set)):
        return type(config)([expand_environment_variables(v) for v in config])
    else:
        return config
