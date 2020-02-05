import importlib
import os
import pkgutil
import sys
from logging import Logger

import pluggy
from pluggy.manager import DistFacade

from . import _builtins, hookspecs

logger = Logger(__name__)

if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    import importlib_metadata


class NapariPluginManager(pluggy.PluginManager):
    PLUGIN_ENTRYPOINT = "napari.plugin"
    PLUGIN_PREFIX = "napari_"

    def __init__(self, autodiscover=True):
        """pluggy.PluginManager subclass with napari-specific functionality

        In addition to the pluggy functionality, this subclass adds
        autodiscovery using package naming convention.

        Parameters
        ----------
        autodiscover : bool or str, optional
            Whether to autodiscover plugins by naming convention and setuptools
            entry_points.  If a string is provided, it is added to sys.path
            before importing, and removed at the end. Any other "truthy" value
            will simply search the current sys.path.  by default True
        """
        super().__init__("napari")

        # define hook specifications and validators
        self.add_hookspecs(hookspecs)

        # register our own built plugins
        self.register(_builtins, name='builtins')
        # discover external plugins
        if not os.environ.get("NAPARI_DISABLE_PLUGIN_AUTOLOAD"):
            if autodiscover:
                self.discover(autodiscover)

    def discover(self, path=None):
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
        int
            The number of modules successfully loaded.
        """
        if path and isinstance(path, str):
            sys.path.insert(0, path)

        count = 0
        if not os.environ.get("NAPARI_DISABLE_ENTRYPOINT_PLUGINS"):
            # register modules defining the napari entry_point in setup.py
            count += self.load_setuptools_entrypoints(self.PLUGIN_ENTRYPOINT)
        if not os.environ.get("NAPARI_DISABLE_NAMEPREFIX_PLUGINS"):
            # register modules using naming convention
            count += self.load_modules_by_prefix(self.PLUGIN_PREFIX)

        if count:
            msg = f'loaded {count} plugins:\n  '
            msg += "\n  ".join([n for n, m in self.list_name_plugin()])
            logger.info(msg)

        if path and isinstance(path, str):
            sys.path.remove(path)

        return count

    def load_setuptools_entrypoints(self, group, name=None):
        """Load modules from querying the specified setuptools ``group``

        Overrides the pluggy method in order to insert try/catch statements.

        Parameters
        ----------
        group : str
            entry point group to load plugins
        name : str, optional
            if given, loads only plugins with the given ``name``.
            by default None

        Returns
        -------
        count : int
            the number of loaded plugins by this call.
        """
        count = 0
        for dist in importlib_metadata.distributions():
            for ep in dist.entry_points:
                if (
                    ep.group != group
                    or (name is not None and ep.name != name)
                    # already registered
                    or self.get_plugin(ep.name)
                    or self.is_blocked(ep.name)
                ):
                    continue
                try:
                    plugin = ep.load()
                    self.register(plugin, name=ep.name)
                    self._plugin_distinfo.append((plugin, DistFacade(dist)))
                except Exception as e:
                    logger.error(
                        f'failed to import plugin: {ep.name}: {str(e)}'
                    )
                    self.unregister(name=ep.name)
                count += 1
        return count

    def load_modules_by_prefix(self, prefix):
        """Find and load modules whose names start with ``prefix``

        Parameters
        ----------
        prefix : str
            The prefix that a module must have in order to be discovered.

        Returns
        -------
        count : int
            The number of modules successfully loaded.
        """
        count = 0
        for finder, name, ispkg in pkgutil.iter_modules():
            if (
                not name.startswith(prefix)
                or self.get_plugin(name)
                or self.is_blocked(name)
            ):
                continue
            try:
                mod = importlib.import_module(name)
                # prevent double registration (e.g. from entry_points)
                if self.is_registered(mod):
                    continue
                self.register(mod, name=name)
                count += 1
            except Exception as e:
                logger.error(f'failed to import plugin: {name}: {str(e)}')
                self.unregister(name=name)
        return count


# for easy availability in try/catch statements without having to import pluggy
# e.g.: except plugin_manager.PluginValidationError
NapariPluginManager.PluginValidationError = (
    pluggy.manager.PluginValidationError
)
