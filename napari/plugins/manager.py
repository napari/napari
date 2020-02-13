import importlib
import os
import pkgutil
import re
import sys
from logging import getLogger
from typing import Optional

import pluggy
from napari import __version__

from . import _builtins, hookspecs

logger = getLogger(__name__)

if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    import importlib_metadata

entry_point_pattern = re.compile(
    r'(?P<module>[\w.]+)\s*'
    r'(:\s*(?P<attr>[\w.]+))?\s*'
    r'(?P<extras>\[.*\])?\s*$'
)


class PluginError(Exception):
    def __init__(self, message, plugin_name=None, plugin_module=None):
        super().__init__(message)
        self.plugin_name = plugin_name
        self.plugin_module = plugin_module


class PluginImportError(PluginError):
    """Raised when a plugin fails to import."""

    def __init__(self, plugin_name, plugin_module):
        msg = f'Failed to import plugin: "{plugin_name}""'
        super().__init__(msg, plugin_name, plugin_module)


class PluginRegistrationError(PluginError):
    """Raised when a plugin fails to register with pluggy."""

    def __init__(self, plugin_name, plugin_module):
        msg = f'Failed to register plugin: "{plugin_name}""'
        super().__init__(msg, plugin_module)


def entry_points_for(group: str):
    for dist in importlib_metadata.distributions():
        for ep in dist.entry_points:
            if ep.group == group:
                yield ep


def modules_starting_with(prefix: str):
    for finder, name, ispkg in pkgutil.iter_modules():
        if name.startswith(prefix):
            yield name


def yield_plugin_modules(
    prefix: Optional[str] = None, group: Optional[str] = None
):
    seen_modules = set()
    if group and not os.environ.get("NAPARI_DISABLE_ENTRYPOINT_PLUGINS"):
        for ep in entry_points_for(group):
            module = entry_point_pattern.match(ep.value).group('module')
            seen_modules.add(module.split(".")[0])
            yield ep.name, module
    if prefix and not os.environ.get("NAPARI_DISABLE_NAMEPREFIX_PLUGINS"):
        for module in modules_starting_with(prefix):
            if module not in seen_modules:
                try:
                    name = importlib_metadata.metadata(module).get('Name')
                except Exception:
                    name = None
                yield name or module, module


def fetch_contact_info(distname):
    try:
        meta = importlib_metadata.metadata(distname)
    except importlib_metadata.PackageNotFoundError:
        return None
    return {
        'name': meta.get('Name'),
        'version': meta.get('Version'),
        'email': meta.get('Author-Email') or meta.get('Maintainer-Email'),
        'url': meta.get('Home-page') or meta.get('Download-Url'),
    }


def log_plugin_error(exc):
    msg = f'\nPluginError: {exc}'
    if exc.__cause__:
        cause = str(exc.__cause__).replace("\n", "\n" + " " * 13)
        msg += f'\n  Cause was: {cause}'
    contact = fetch_contact_info(exc.plugin_module)
    if contact:
        msg += "\n  Please notify the plugin developer:\n"
        extra = [f'{k: >11}: {v}' for k, v in contact.items()]
        extra += [f'{"napari": >11}: v{__version__}']
        msg += "\n".join(extra)
    logger.error(msg)


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
        for plugin_name, module in yield_plugin_modules(
            prefix=self.PLUGIN_PREFIX, group=self.PLUGIN_ENTRYPOINT
        ):
            if self.get_plugin(plugin_name) or self.is_blocked(plugin_name):
                continue
            try:
                try:
                    mod = importlib.import_module(module)
                except Exception as exc:
                    raise PluginImportError(plugin_name, module) from exc
                try:
                    # prevent double registration (e.g. from entry_points)
                    if self.is_registered(mod):
                        continue
                    self.register(mod, name=plugin_name)
                except Exception as exc:
                    raise PluginRegistrationError(plugin_name, module) from exc
                count += 1
            except PluginError as exc:
                log_plugin_error(exc)
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

        if path and isinstance(path, str):
            sys.path.remove(path)

        return count


# for easy availability in try/catch statements without having to import pluggy
# e.g.: except plugin_manager.PluginValidationError
NapariPluginManager.PluginValidationError = (
    pluggy.manager.PluginValidationError
)
