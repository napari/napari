"""napari configuration

This module is copied, with modifications, from dask.config

We like Dask's "plain python dict" treatment of configurations, handling of
nested namespaces, backwards compatilibity of new settings, context management
(temporary setting) of config values, choice of yaml for persistence, and
environment variable options

see documentation for dask.config at:
https://docs.dask.org/en/latest/configuration.html

Configuration is specified in one of the following ways:

1. YAML files in ~/.config/napari/ or /etc/napari/
2. Environment variables that start with ``NAPARI_``
3. Default settings within sub-libraries

This combination makes it easy to specify configuration in a variety of
settings ranging from personal workstations, to IT-mandated configuration, to
docker images.

Loading
-------

When this module is imported, this sequence of events happens:

1. the config is cleared and ``refresh()`` is called
2. ``refresh()`` updates the empty config with values from dicts in the
   `config.defaults` list (currently empty)
3. ``refresh()`` then updates the config by calling ``collect()`` ...
   a. ``collect()`` gathers settings from various places:
      - ``collect_yaml()`` looks for *any* `.yaml` files in the
         ``config.paths`` list.  Notably, this includes
         ``~/.config/napari/napari.yaml`` and anything else in there.
      - ``collec_env()`` then looks for any environment variables that begin
         with ``NAPARI_``
   b. finally, ``collect()`` calls `merge()` on all of those discovered dicts,
      to yield one merged result and returns it to the ``refresh()`` function
      (by default, dicts discovered later in the process take priority).
4. back in the ``refresh()`` function, that final merged result is used to
   update the global config dict.
5. At that point, the ``napari.yaml`` in the repo is read, if one doesn't
   already exist, a file is created at ``~/.config/napari/napari.yaml`` (with
   ``ensure_file()``) that is an exact copy of the repo ``napari.yaml`` but
   with everything commented out, as a way to show users what they can change.
6. finally, ``update_defaults()`` is called using the settings in the
   `napari.yaml` file, which merges the configs again, but gives priority to
   already existing values in the config (i.e. ones that were overwritten by
   the user or the environement)

Writing
-------

To write settings to disk, the :func:`sync` function is used.  This
synchronizes the current ``config`` dict to a file at
``join(config.PATH, 'napari.yaml')``.  If that file has changed since the last
sync, it also reads new values from that file into the config.  If *both*
the file and the config have changed, it prioritizes values in the config.
"""
import ast
import logging
import os
import sys
import threading
from collections import defaultdict
from functools import wraps
from types import TracebackType
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import yaml

from ..utils.misc import StringEnum
from .deprecations import check_deprecations
from .utils import canonical_name, ensure_file, merge, update

#: a list of paths that get searched for .yaml files in :func:`collect_yaml`
paths = [
    os.getenv("NAPARI_ROOT_CONFIG", "/etc/napari"),
    os.path.join(sys.prefix, "etc", "napari"),
    os.path.join(os.path.expanduser("~"), ".config", "napari"),
    os.path.join(os.path.expanduser("~"), ".napari"),
]


#: The primary directory for user config files
#:
#: Defaults to ~/.config/napari
#: May be set with the environment variable "NAPARI_CONFIG"
PATH = os.path.join(os.path.expanduser("~"), ".config", "napari")
if "NAPARI_CONFIG" in os.environ:
    PATH = os.environ["NAPARI_CONFIG"]
    paths.append(PATH)

#: a special yaml file to dump settings that are changed within a session
#:
#: this file will be loaded, but takes lower priority than all non-private
#: yaml files.  So if a user has a specific setting in their napari.yaml file,
#: it will override anything in _session.yaml
_SESSION = os.path.join(PATH, '_session.yaml')


#: the main global config dict
config: Dict[str, Any] = {}

CallbackDict = DefaultDict[Tuple[str, ...], Set[Callable[[Any], None]]]
#: mapping from key (as tuple of strings), to callback function
#:
#: callback will be called whenever the key changes
callbacks: CallbackDict = defaultdict(set)

config_lock = threading.Lock()

#: A list of config dicts that downstream libraries may use to register defaults
#:
#: see :func:`update_defaults` and
#: https://docs.dask.org/en/latest/configuration.html#downstream-libraries
defaults: List[dict] = []

no_default = "__no_default__"


def get(key: str, default=no_default, config: dict = config) -> Any:
    """Get elements from global config.

    Use '.' for nested access

    Examples
    --------
    >>> from napari import config
    >>> config.get('foo')
    {'x': 1, 'y': 2}

    >>> config.get('foo.x')
    1

    >>> config.get('foo.x.y', default=123)
    123

    See Also
    --------
    napari.config.set
    """
    keys = key.split(".")
    result = config
    for k in keys:
        k = canonical_name(k, result)
        try:
            result = result[k]
        except (TypeError, IndexError, KeyError):
            if default is not no_default:
                return default
            else:
                raise
    return result


def pop(key: str, default=no_default, config: dict = config):
    """Pop elements from global config.

    Use '.' for nested access

    By default, the key `_dirty` is added to the config dict so that the
    :func:`sync` function knows to update the yaml file on disk.

    Examples
    --------
    >>> from napari import config
    >>> config.set({'foo': {'a': 1, 'b': 2}})
    >>> config.pop('foo.b')
    2

    >>> config.pop('foo')
    {'a': 1}

    >>> config.pop('foo.x.y', default=123)
    123

    See Also
    --------
    napari.config.get
    """
    keys = tuple(key.split("."))

    result = config
    for i, k in enumerate(keys):
        k = canonical_name(k, result)
        try:
            if i == len(keys) - 1:
                result = result.pop(k)
                config['_dirty'] = True
                _callback_to_listeners(keys, None, callbacks, result)

            else:
                result = result[k]
        except (TypeError, IndexError, KeyError):
            if default is not no_default:
                return default
            else:
                raise
    return result


class _set:
    """Temporarily set configuration values within a context manager.

    Parameters
    ----------
    arg : mapping or None, optional
        A mapping of configuration key-value pairs to set.
    **kwargs :
        Additional key-value pairs to set. If ``arg`` is provided, values set
        in ``arg`` will be applied before those in ``kwargs``.
        Double-underscores (``__``) in keyword arguments will be replaced with
        ``.``, allowing nested values to be easily set.

    Examples
    --------
    >>> import napari

    Set ``'foo.bar'`` in a context, by providing a mapping.

    >>> with napari.config.set({'foo.bar': 123}):
    ...     pass

    Set ``'foo.bar'`` in a context, by providing a keyword argument.

    >>> with napari.config.set(foo__bar=123):
    ...     pass

    Set ``'foo.bar'`` globally.

    >>> napari.config.set(foo__bar=123)

    See Also
    --------
    napari.config.get
    """

    def __init__(
        self,
        arg: Optional[dict] = None,
        config: dict = config,
        lock: threading.Lock = config_lock,
        callbacks: CallbackDict = callbacks,
        **kwargs,
    ):
        with lock:
            self.config = config
            self.callbacks = callbacks
            self._record: List[Tuple[str, Tuple[str, ...], Any]] = []

            if arg is not None:
                try:
                    for key, value in arg.items():
                        key = check_deprecations(key)
                        self._assign(key.split("."), value, config)
                except AttributeError:
                    if not isinstance(arg, dict):
                        raise TypeError(
                            "First argument to config.set() must be a dict"
                        )
                    else:
                        raise

            if kwargs:
                for key, value in kwargs.items():
                    key = key.replace("__", ".")
                    key = check_deprecations(key)
                    self._assign(key.split("."), value, config)

        config['_dirty'] = True

    def __enter__(self) -> dict:
        return self.config

    def __exit__(
        self,
        type: Type[Exception],
        value: Exception,
        traceback: Optional[TracebackType],
    ):
        for op, path, value in reversed(self._record):
            d = self.config
            old_val = None
            if op == "replace":
                for key in path[:-1]:
                    d = d.setdefault(key, {})
                old_val = d[path[-1]]
                d[path[-1]] = value
            else:  # insert
                for key in path[:-1]:
                    try:
                        d = d[key]
                    except KeyError:
                        break
                else:
                    old_val = d.pop(path[-1], None)
            _callback_to_listeners(path, value, self.callbacks, old_val)

    def _assign(
        self,
        keys: Sequence[str],
        value: Any,
        d: dict,
        path: Tuple[str, ...] = (),
        record: bool = True,
    ):
        """Assign value into a nested configuration dictionary

        Parameters
        ----------
        keys : Sequence[str]
            The nested path of keys to assign the value.
        value : object
        d : dict
            The part of the nested dictionary into which we want to assign the
            value
        path : Tuple[str], optional
            The path history up to this point.
        record : bool, optional
            Whether this operation needs to be recorded to allow for rollback.
        """
        key = canonical_name(keys[0], d)

        path = path + (key,)

        if len(keys) == 1:
            old_val = None
            if record:
                if key in d:
                    self._record.append(("replace", path, d[key]))
                    old_val = d[key]
                else:
                    self._record.append(("insert", path, None))
            d[key] = value
            _callback_to_listeners(path, value, self.callbacks, old_val)

            # might be worth emitting a warning here if we know the value
            # cannot be serialized
            # yaml.dump(value, Dumper=ConfigDumper)

        else:
            if key not in d:
                if record:
                    self._record.append(("insert", path, None))
                d[key] = {}
                # No need to record subsequent operations after an insert
                record = False
            self._assign(keys[1:], value, d[key], path, record=record)


def collect_yaml(paths: List[str] = paths) -> List[dict]:
    """Collect configuration from yaml files.

    This searches through a list of paths, expands to find all yaml or json
    files, and then parses each file.
    """
    # Find all paths
    file_paths = []
    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                try:
                    for fname in sorted(os.listdir(path)):
                        base, ext = os.path.splitext(fname)
                        if base.startswith("_"):
                            continue
                        if ext.lower() in (".json", ".yaml", ".yml"):
                            file_paths.append(os.path.join(path, fname))
                except OSError:
                    # Ignore permission errors
                    pass
            else:
                file_paths.append(path)
    configs = []

    # Parse yaml files
    for path in file_paths:
        try:
            with open(path) as f:
                data = yaml.safe_load(f.read()) or {}
                configs.append(data)
        except (OSError, IOError):
            # Ignore permission errors
            pass

    return configs


def collect_env(env: Optional[Union[dict, os._Environ]] = None) -> dict:
    """Collect config from environment variables.

    This grabs environment variables of the form "NAPARI_FOO__BAR_BAZ=123" and
    turns these into config variables of the form ``{"foo": {"bar-baz": 123}}``
    It transforms the key and value in the following way:

    -  Lower-cases the key text
    -  Treats ``__`` (double-underscore) as nested access
    -  Calls ``ast.literal_eval`` on the value
    """
    if env is None:
        env = os.environ
    d = {}
    for name, value in env.items():
        if name.startswith("NAPARI_"):
            varname = name[7:].lower().replace("__", ".")
            try:
                d[varname] = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                d[varname] = value

    result: dict = {}
    _set(d, config=result)

    return result


def collect(
    paths: List = paths, env: Optional[Union[dict, os._Environ]] = None
) -> dict:
    """
    Collect configuration from paths and environment variables

    Parameters
    ----------
    paths : List[str]
        A list of paths to search for yaml config files

    env : dict
        The system environment variables

    Returns
    -------
    config: dict

    See Also
    --------
    napari.config.refresh: collect configuration and update into primary config
    """
    if env is None:
        env = os.environ

    configs = []
    configs.extend(collect_yaml(paths=paths))
    configs.append(collect_env(env=env))

    return merge(*configs)


def refresh(config: dict = config, defaults: List[dict] = defaults, **kwargs):
    """
    Update configuration by re-reading yaml files and env variables

    This mutates the global napari.config.config, or the config parameter if
    passed in.

    This goes through the following stages:

    1.  Clearing out all old configuration
    2.  Updating from the stored defaults from downstream libraries
        (see update_defaults)
    3.  Updating from yaml files and environment variables

    Note that some functionality only checks configuration once at startup and
    may not change behavior, even if configuration changes.  It is recommended
    to restart your python process if convenient to ensure that new
    configuration changes take place.

    See Also
    --------
    napari.config.collect: for parameters
    napari.config.update_defaults
    """
    config.clear()

    for d in defaults:
        update(config, d, priority="old")

    update(config, collect(**kwargs))


def rename(aliases: dict, config: dict = config):
    """Rename old keys to new keys.

    This helps migrate older configuration versions over time
    """
    old = []
    new = {}
    for o, n in aliases.items():
        value = get(o, None, config=config)
        if value is not None:
            old.append(o)
            new[n] = value

    for k in old:
        del config[canonical_name(k, config)]  # TODO: support nested keys

    _set(new, config=config)


def update_defaults(
    new: dict, config: dict = config, defaults: List[dict] = defaults
):
    """Add a new set of defaults to the configuration.

    Used internally, but also intended for use by downstream libraries. See:
    https://docs.dask.org/en/latest/configuration.html#downstream-libraries

    It does two things:

    1. Add the defaults to a global ``defaults`` collection to be used by
       refresh later
    2. Updates the global config with the new configurationm, prioritizing
       older values over newer ones
    """
    defaults.append(new)
    update(config, new, priority="old")


def register_listener(
    key: Union[str, Tuple[str, ...]],
    callback: Callable[[Any], None],
    callbacks: CallbackDict = callbacks,
):
    path: Tuple = tuple(key.split('.')) if isinstance(key, str) else key
    if not callable(callback):
        raise ValueError(
            "'callback' must be a function that accepts one parameter"
        )
    callbacks[path].add(callback)


def _callback_to_listeners(
    path: Tuple[str, ...],
    new_value: Any,
    callbacks: CallbackDict = callbacks,
    old_val=None,
):
    if old_val is not None and old_val == new_value:
        return
    if path in callbacks:
        for cb in callbacks[path]:
            cb(new_value)


def updates_config(
    key: str, config: dict = config
) -> Callable[[Callable], Callable]:
    """Return decorator that updates config when decorated function is called.

    Parameters
    ----------
    key : str
        the config key to update
    config : dict, optional
        The config to update, by default updates global config
    """

    def decorator(function):
        """Return function that updates config when original func is called."""

        @wraps(function)
        def modified_function(*args):
            function(*args)
            # assume last item in *args is the new value
            # works for both method `setter(self, val)` and func `setter(val)`
            if get(key, None, config=config) != args[-1]:
                _set({key: args[-1]}, config=config)

        return modified_function

    return decorator


class ConfigDumper(yaml.SafeDumper):
    """Dumper that prevents yaml aliases, and logs bad objects without error.
    """

    def ignore_aliases(self, data):
        return True

    def represent_undefined(self, data):
        logging.error("Error serializing object: %r", data)
        return self.represent_str('<unserializeable>')

    def coerce_to_str(self, data) -> yaml.nodes.ScalarNode:
        return self.represent_str(str(data))


ConfigDumper.add_multi_representer(StringEnum, ConfigDumper.coerce_to_str)
ConfigDumper.add_multi_representer(None, ConfigDumper.represent_undefined)


def sync(
    config: dict = config,
    destination: str = None,
    lock: threading.Lock = config_lock,
) -> bool:
    """Synchronize config with a yaml file on disk.

    This function is intended to be run periodically in the background of an
    event loop.  It looks for a special ``_dirty`` in the config to know
    whether it has changed since the last time this function was called (and
    pops that key when this function runs).  It also looks for a
    ``_last_synced`` key, which should contain a float value corresponding to
    the last time this function was called; if the modification time of the
    destination file is greater than ``_last_synced``, it is assumed that the
    yaml on disk has changed.

    Parameters
    ----------
    config : dict, optional
        The config to sync to disk, by default use the global config
    destination : str, optional
        Filename or directory to sync to, by default will sync to
        ``config.PATH/_session.yaml``
    lock : threading.Lock, optional
        A threading.Lock instance to protect read/write on ``destination``,
        by default, this module's config_lock is used.

    Returns
    -------
    synced : bool
        Whether a sync occurred.
    """
    if not destination:
        from . import _SESSION

        destination = _SESSION

    if not os.fspath(destination).endswith((".yaml", ".yml")):
        raise ValueError("Only YAML is currently supported")

    config_is_dirty = config.pop('_dirty', None)
    if not os.path.exists(destination):
        config_is_dirty = True
    # do nothing the config hasn't changed
    if not config_is_dirty:
        return False

    with lock:  # aquire file lock on yaml file
        # write the config to disk
        with open(destination, 'w') as f:
            if config:
                try:
                    yaml.dump(config, f, Dumper=ConfigDumper)
                except yaml.YAMLError as exc:
                    msg = f"Failed to write session config to disk: {exc}"
                    raise type(exc)(msg)
            else:  # instead of writing "{}" to file, write "# empty"
                f.write("# empty")

    return True


def initialize():

    # clear out config and initialize with yaml files from the user directory,
    # env variables, and downstream libraries that have used update_defaults()
    refresh()

    # read in the default settings from this directory
    napari_defaults = os.path.join(os.path.dirname(__file__), "napari.yaml")
    # and make sure it exists (commented out) in the user config dir
    ensure_file(source=napari_defaults)

    # add our internal defaults to the config
    with open(napari_defaults) as f:
        update_defaults(yaml.safe_load(f) or {})

    # if a session file from the last session is available load it, but do not
    # overwrite any settings declared in one of the main config yaml files.
    if os.path.isfile(_SESSION):
        with open(_SESSION) as f:
            update(config, yaml.safe_load(f) or {}, priority="old")

    config.pop("_dirty", None)
