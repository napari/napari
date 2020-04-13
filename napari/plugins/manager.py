import importlib
import os
import pkgutil
import sys
from logging import getLogger
from typing import Dict, Generator, Optional, Tuple, Union

import pluggy

from . import _builtins, hook_specifications
from ._hook_callers import _HookCaller
from .exceptions import (
    PluginError,
    PluginImportError,
    PluginRegistrationError,
    fetch_module_metadata,
)

logger = getLogger(__name__)

if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    import importlib_metadata


pluggy.manager._HookCaller = _HookCaller


class _HookRelay:
    """Hook holder object for storing _HookCaller instances.

    This object triggers (lazy) discovery of plugins as follows:  When a plugin
    hook is accessed (e.g. plugin_manager.hook.napari_get_reader), if
    ``self._needs_discovery`` is True, then it will trigger autodiscovery on
    the parent plugin_manager. Note that ``PluginManager.__init__`` sets
    ``self.hook._needs_discovery = True`` *after* hook_specifications and
    builtins have been discovered, but before external plugins are loaded.
    """

    _needs_discovery = False

    def __init__(self, manager: 'PluginManager'):
        self._manager = manager

    def __getattribute__(self, name):
        if name not in ('_needs_discovery', '_manager'):
            if self._needs_discovery:
                self._needs_discovery = False
                self._manager.discover()
        return object.__getattribute__(self, name)


class PluginManager(pluggy.PluginManager):
    PLUGIN_ENTRYPOINT = "napari.plugin"
    PLUGIN_PREFIX = "napari_"

    def __init__(
        self,
        project_name: str = "napari",
        autodiscover: Union[bool, str] = False,
    ):
        """pluggy.PluginManager subclass with napari-specific functionality

        In addition to the pluggy functionality, this subclass adds
        autodiscovery using package naming convention.

        Parameters
        ----------
        project_name : str, optional
            Namespace for plugins managed by this manager. by default 'napari'.
        autodiscover : bool or str, optional
            Whether to autodiscover plugins by naming convention and setuptools
            entry_points.  If a string is provided, it is added to sys.path
            before importing, and removed at the end. Any other "truthy" value
            will simply search the current sys.path.  by default True
        """
        super().__init__(project_name)
        self.hook = _HookRelay(self)
        # a dict to store package metadata for each plugin, will be populated
        # during self._register_module
        # possible keys for this dict will be set by fetch_module_metadata()
        self._plugin_meta: Dict[str, Dict[str, str]] = dict()

        # project_name might not be napari if running tests
        if project_name == 'napari':
            # define hook specifications and validators
            self.add_hookspecs(hook_specifications)
            # register our own builtin plugins
            self.register(_builtins, name='builtins')

        self.hook._needs_discovery = True
        # discover external plugins
        if autodiscover:
            if isinstance(autodiscover, str):
                self.discover(autodiscover)
            else:
                self.discover()

    @property
    def hooks(self):
        """An alias for PluginManager.hook"""
        return self.hook

    def discover(self, path: Optional[str] = None) -> int:
        """Discover modules by both naming convention and entry_points

        1) Using naming convention:
            plugins installed in the environment that follow a naming
            convention (e.g. "napari_plugin"), can be discovered using
            `pkgutil`. This also enables easy discovery on pypi

        2) Using package metadata:
            plugins that declare a special key (self.PLUGIN_ENTRYPOINT) in
            their setup.py `entry_points`.  discovered using `pkg_resources`.

        https://packaging.python.org/guides/creating-and-discovering-plugins/

        Parameters
        ----------
        path : str, optional
            If a string is provided, it is added to sys.path before importing,
            and removed at the end. by default True

        Returns
        -------
        count : int
            The number of plugin modules successfully loaded.
        """
        if path is None:
            self.hook._needs_discovery = False

        # allow debugging escape hatch
        if os.environ.get("NAPARI_DISABLE_PLUGINS"):
            import warnings

            warnings.warn(
                'Plugin discovery disabled due to '
                'environmental variable "NAPARI_DISABLE_PLUGINS"'
            )
            return 0

        if path:
            sys.path.insert(0, path)

        count = 0
        for plugin_name, module_name, meta in iter_plugin_modules(
            prefix=self.PLUGIN_PREFIX, group=self.PLUGIN_ENTRYPOINT
        ):
            if self.get_plugin(plugin_name) or self.is_blocked(plugin_name):
                continue
            try:
                self._register_module(plugin_name, module_name, meta)
                count += 1
            except PluginError as exc:
                logger.error(exc.format_with_contact_info())
                self.unregister(name=plugin_name)
            except Exception as exc:
                logger.error(
                    f'Unexpected error loading plugin "{plugin_name}": {exc}'
                )
                self.unregister(name=plugin_name)

        if count:
            msg = f'loaded {count} plugins:\n  '
            msg += "\n  ".join([n for n, m in self.list_name_plugin()])
            logger.info(msg)

        if path:
            sys.path.remove(path)

        return count

    def _register_module(
        self, plugin_name: str, module_name: str, meta: Optional[dict] = None
    ):
        """Try to register `module_name` as a plugin named `plugin_name`.

        Parameters
        ----------
        plugin_name : str
            The name given to the plugin in the plugin manager.
        module_name : str
            The importable module name
        meta : dict, optional
            Metadata to be associated with ``plugin_name``.

        Raises
        ------
        PluginImportError
            If an error is raised when trying to import `module_name`
        PluginRegistrationError
            If an error is raised when trying to register the plugin (such as
            a PluginValidationError.)
        """
        if meta:
            meta.update({'plugin': plugin_name})
            self._plugin_meta[plugin_name] = meta
        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            raise PluginImportError(plugin_name, module_name) from exc
        try:
            # prevent double registration (e.g. from entry_points)
            if self.is_registered(mod):
                return
            self.register(mod, name=plugin_name)
        except Exception as exc:
            raise PluginRegistrationError(plugin_name, module_name) from exc


def entry_points_for(
    group: str,
) -> Generator[
    Tuple[importlib_metadata.Distribution, importlib_metadata.EntryPoint],
    None,
    None,
]:
    """Yield all entry_points matching "group", from any distribution.

    Distribution here refers more specifically to the information in the
    dist-info folder that usually accompanies an installed package.  If a
    package in the environment does *not* have a ``dist-info/entry_points.txt``
    file, then it will not be discovered by this function.

    Note: a single package may provide multiple entrypoints for a given group.

    Parameters
    ----------
    group : str
        The name of the entry point to search.

    Yields
    -------
    tuples
        (Distribution, EntryPoint) objects for each matching EntryPoint
        that matches the provided ``group`` string.

    Example
    -------
    >>> list(entry_points_for('napari.plugin'))
    [(<importlib.metadata.PathDistribution at 0x124f0fe80>,
      EntryPoint(name='napari-reg',value='napari_reg',group='napari.plugin')),
     (<importlib.metadata.PathDistribution at 0x1041485b0>,
      EntryPoint(name='myplug',value='another.module',group='napari.plugin'))]
    """
    for dist in importlib_metadata.distributions():
        for ep in dist.entry_points:
            if ep.group == group:
                yield dist, ep


def modules_starting_with(prefix: str) -> Generator[str, None, None]:
    """Yield all module names in sys.path that begin with `prefix`.

    Parameters
    ----------
    prefix : str
        The prefix to search

    Yields
    -------
    module_name : str
        Yields names of modules that start with prefix

    """
    for finder, name, ispkg in pkgutil.iter_modules():
        if name.startswith(prefix):
            yield name


def iter_plugin_modules(
    prefix: Optional[str] = None, group: Optional[str] = None
) -> Generator[Tuple[str, str, dict], None, None]:
    """Discover plugins using naming convention and/or entry points.

    This function makes sure that packages that *both* follow the naming
    convention (i.e. starting with `prefix`) *and* provide and an entry point
    `group` will only be yielded once.  Precedence is given to entry points:
    that is, if a package satisfies both critera, only the modules specifically
    listed in the entry points will be yielded.  These MAY or MAY NOT be the
    top level module in the package... whereas with naming convention, it is
    always the top level module that gets imported and registered with the
    plugin manager.

    The NAME of yielded plugins will be the name of the package provided in
    the package METADATA file when found.  This allows for the possibility that
    the plugin name and the module name are not the same: for instance...
    ("napari-plugin", "napari_plugin").

    Plugin packages may also provide multiple entry points, which will be
    registered as plugins of different names.  For instance, the following
    ``setup.py`` entry would register two plugins under the names
    ``myplugin.register`` and ``myplugin.segment``

    .. code-block:: python

        import sys

        setup(
            name="napari-plugin",
            entry_points={
                "napari.plugin": [
                    "myplugin.register = napari_plugin.registration",
                    "myplugin.segment = napari_plugin.segmentation"
                ],
            },
            packages=find_packages(),
        )


    Parameters
    ----------
    prefix : str, optional
        A prefix by which to search module names.  If None, discovery by naming
        convention is disabled., by default None
    group : str, optional
        An entry point group string to search.  If None, discovery by Entry
        Points is disabled, by default None

    Yields
    -------
    plugin_info : tuple
        (plugin_name, module_name, metadata)
    """
    seen_modules = set()
    if group and not os.environ.get("NAPARI_DISABLE_ENTRYPOINT_PLUGINS"):
        for dist, ep in entry_points_for(group):
            match = ep.pattern.match(ep.value)
            if match:
                module = match.group('module')
                seen_modules.add(module.split(".")[0])
                yield ep.name, module, fetch_module_metadata(dist)
    if prefix and not os.environ.get("NAPARI_DISABLE_NAMEPREFIX_PLUGINS"):
        for module in modules_starting_with(prefix):
            if module not in seen_modules:
                try:
                    name = importlib_metadata.metadata(module).get('Name')
                except Exception:
                    name = None
                yield name or module, module, fetch_module_metadata(module)
