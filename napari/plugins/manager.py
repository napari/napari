import importlib
import os
import pkgutil
from logging import Logger
from types import ModuleType
from typing import List, Union

import pluggy

from . import _builtins, hookspecs

logger = Logger(__name__)


class NapariPluginManager(pluggy.PluginManager):
    PLUGIN_ENTRYPOINT = "napari.plugin"
    PLUGIN_PREFIX = "napari_"

    def __init__(self, autodiscover=True):
        """pluggy.PluginManager subclass with napari-specific functionality

        In addition to the pluggy functionality, this subclass adds
        autodiscovery using package naming convention.

        Parameters
        ----------
        autodiscover : bool, optional
            Whether to autodiscover plugins by naming convention and setuptools
            entry_points, by default True
        """
        super().__init__("napari")

        # define hook specifications and validators
        self.add_hookspecs(hookspecs)

        # register our own built plugins
        self.register(_builtins, name='builtins')
        # discover external plugins
        if not os.environ.get("NAPARI_DISABLE_PLUGIN_AUTOLOAD"):
            if autodiscover:
                self.discover()

    def discover(self):
        """Discover modules by both naming convention and entry_points

        1) Using naming convention:
            plugins installed in the environment that follow a naming
            convention (e.g. "napari_plugin"), can be discovered using
            `pkgutil`. This also enables easy discovery on pypi

        2) Using package metadata:
            plugins that declare a special key (self.PLUGIN_ENTRYPOINT) in
            their setup.py `entry_points`.  discovered using `pkg_resources`.

        https://packaging.python.org/guides/creating-and-discovering-plugins/

        Returns
        -------
        int
            The number of modules successfully loaded.
        """
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
        return count

    def load_modules_by_prefix(self, prefix):
        """Find and load modules whose names start with ``prefix``

        Parameters
        ----------
        prefix : str
            The prefix that a module must have in order to be discovered.

        Returns
        -------
        int
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
                self.unregister(mod)
        return count


# for easy try/catch availability
NapariPluginManager.PluginValidationError = (
    pluggy.manager.PluginValidationError
)


def permute_hookimpls(
    hook_caller: pluggy.hooks._HookCaller,
    order: Union[List[str], List[ModuleType], List[pluggy.hooks.HookImpl]],
):
    """Change the call order of hookimplementations for a pluggy HookCaller.

    Pluggy does not allow a built-in way to change the call order after
    instantiation.  hookimpls are called in last-in-first-out order.
    This function accepts the desired call order (a list of plugin names, or
    plugin modules) and reorders the hookcaller accordingly.

    Parameters
    ----------
    hook_caller : pluggy.hooks._HookCaller
        The hook caller to reorder
    order : list
        A list of str, hookimpls, or module_or_class, with the desired
        CALL ORDER of the hook implementations.

    Raises
    ------
    ValueError
        If the 'order' list cannot be interpreted as a list of "plugin_name"
        or "plugin" (module_or_class)
    ValueError
        if 'order' argument has multiple entries for the same hookimpl
    """
    if all(isinstance(o, pluggy.hooks.HookImpl) for o in order):
        attr = None
    elif all(isinstance(o, str) for o in order):
        attr = 'plugin_name'
    elif any(isinstance(o, str) for o in order):
        raise TypeError(
            "order list must be either ALL strings, or ALL modules/classes"
        )
    else:
        attr = 'plugin'

    hookimpls = hook_caller.get_hookimpls()
    if len(order) > len(hookimpls):
        raise ValueError(
            f"too many values ({len(order)} > {len(hookimpls)}) in order."
        )
    if attr:
        hookattrs = [getattr(hookimpl, attr) for hookimpl in hookimpls]
    else:
        hookattrs = hookimpls

    # find the current position of items specified in `order`
    indices = []
    seen = set()
    for i in order:
        if i in seen:
            raise ValueError(
                f"'order' argument had multiple entries for hookimpl: {i}"
            )
        seen.add(i)
        try:
            indices.append(hookattrs.index(i))
        except ValueError as e:
            msg = f"Could not find hookimpl '{i}'."
            if attr != 'plugin_name':
                msg += (
                    " If all items in `order` "
                    "argument are not strings, they are assumed to be an "
                    "imported plugin module or class."
                )
            raise ValueError(msg) from e

    # make new arrays for _wrappers and _nonwrappers
    _wrappers = []
    _nonwraps = []
    for i in indices:
        imp = hookimpls[i]
        methods = _wrappers if imp.hookwrapper else _nonwraps
        methods.insert(0, imp)

    # remove items that have been pulled, leaving only items that
    # were not specified in `order` argument
    for i in sorted(indices, reverse=True):
        del hookimpls[i]

    if hookimpls:
        _wrappers = [x for x in hookimpls if x.hookwrapper] + _wrappers
        _nonwraps = [x for x in hookimpls if not x.hookwrapper] + _nonwraps

    hook_caller._wrappers = _wrappers
    hook_caller._nonwrappers = _nonwraps
