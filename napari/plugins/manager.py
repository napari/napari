import importlib
import os
import pkgutil
import re
import sys
from logging import getLogger
from typing import Generator, Optional, Tuple, Union, List

import pluggy
from pluggy.hooks import HookImpl

from . import _builtins, hook_specifications
from .exceptions import PluginError, PluginImportError, PluginRegistrationError

logger = getLogger(__name__)

if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    import importlib_metadata


class _HookCaller(pluggy.hooks._HookCaller):
    """Adding convenience methods to PluginManager.hook

    In a pluggy plugin manager, the hook implementations registered for each
    plugin are stored in ``_HookCaller`` objects that share the same name as
    the corresponding hook specification; and each ``_HookCaller`` instance is
    stored under the ``plugin_manager.hook`` namespace. For instance:
    ``plugin_manager.hook.name_of_hook_specification``.
    """

    # just for type annotation.  These are the lists that store HookImpls
    _wrappers: List[HookImpl]
    _nonwrappers: List[HookImpl]

    def get_hookimpl_for_plugin(self, plugin_name: str):
        """Return hook implementation instance for ``plugin_name`` if found."""
        try:
            return next(
                imp
                for imp in self.get_hookimpls()
                if imp.plugin_name == plugin_name
            )
        except StopIteration:
            raise KeyError(
                f"No implementation of {self.name} found "
                f"for plugin {plugin_name}."
            )

    def index(self, value: Union[str, HookImpl]) -> int:
        """Return index of plugin_name or a HookImpl in self._nonwrappers"""
        if isinstance(value, HookImpl):
            return self._nonwrappers.index(value)
        elif isinstance(value, str):
            plugin_names = [imp.plugin_name for imp in self._nonwrappers]
            return plugin_names.index(value)
        else:
            raise TypeError(
                "argument provided to index must either be the "
                "(string) name of a plugin, or a HookImpl instance"
            )

    def bring_to_front(self, new_order: Union[List[str], List[HookImpl]]):
        """Move items in ``new_order`` to the front of the call order.

        By default, hook implementations are called in last-in-first-out order
        of registration, and pluggy does not provide a built-in way to
        rearrange the call order of hook implementations.

        This function accepts a `_HookCaller` instance and the desired
        ``new_order`` of the hook implementations (in the form of list of
        plugin names, or a list of actual ``HookImpl`` instances) and reorders
        the implementations in the hook caller accordingly.

        NOTE: hook implementations are actually stored in *two* separate list
        attributes in the hook caller: ``_HookCaller._wrappers`` and
        ``_HookCaller._nonwrappers``, according to whether the corresponding
        ``HookImpl`` instance was marked as a wrapper or not.  This method
        *only* sorts _nonwrappers.
        For more, see: https://pluggy.readthedocs.io/en/latest/#wrappers

        Parameters
        ----------
        new_order :  list of str or list of ``HookImpl`` instances
            The desired CALL ORDER of the hook implementations.  The list
            does *not* need to include every hook implementation in
            ``self.get_hookimpls()``, but those that are not included
            will be left at the end of the call order.

        Raises
        ------
        TypeError
            If any item in ``new_order`` is neither a string (plugin_name) or a
            ``HookImpl`` instance.
        ValueError
            If any item in ``new_order`` is neither the name of a plugin or a
            ``HookImpl`` instance that is present in self._nonwrappers.
        ValueError
            If ``new_order`` argument has multiple entries for the same
            implementation.

        Examples
        --------
        Imagine you had a hook specification named ``print_plugin_name``, that
        expected plugins to simply print their own name. An implementation
        might look like:

        >>> # hook implementation for ``plugin_1``
        >>> @hook_implementation
        ... def print_plugin_name():
        ...     print("plugin_1")

        If three different plugins provided hook implementations. An example
        call for that hook might look like:

        >>> plugin_manager.hook.print_plugin_name()
        plugin_1
        plugin_2
        plugin_3

        If you wanted to rearrange their call order, you could do this:

        >>> new_order = ["plugin_2", "plugin_3", "plugin_1"]
        >>> plugin_manager.hook.print_plugin_name.bring_to_front(new_order)
        >>> plugin_manager.hook.print_plugin_name()
        plugin_2
        plugin_3
        plugin_1

        You can also just specify one or more item to move them to the front
        of the call order:
        >>> plugin_manager.hook.print_plugin_name.bring_to_front(["plugin_3"])
        >>> plugin_manager.hook.print_plugin_name()
        plugin_3
        plugin_2
        plugin_1
        """
        # make sure items in order are unique
        if len(new_order) != len(set(new_order)):
            raise ValueError("repeated item in order")

        # make new lists for the rearranged _nonwrappers
        # for details on the difference between wrappers and nonwrappers, see:
        # https://pluggy.readthedocs.io/en/latest/#wrappers
        _old_nonwrappers = self._nonwrappers.copy()
        _new_nonwrappers: List[HookImpl] = []
        indices = [self.index(elem) for elem in new_order]
        for i in indices:
            _new_nonwrappers.insert(0, _old_nonwrappers[i])

        # remove items that have been pulled, leaving only items that
        # were not specified in ``new_order`` argument
        # do this rather than using .pop() above to avoid changing indices
        for i in sorted(indices, reverse=True):
            del _old_nonwrappers[i]

        # if there are any hook_implementations left over, add them to the
        # beginning of their respective lists
        if _old_nonwrappers:
            _new_nonwrappers = [x for x in _old_nonwrappers] + _new_nonwrappers

        # update the _nonwrappers list with the reordered list
        self._nonwrappers = _new_nonwrappers

    def _set_plugin_enabled(self, plugin_name: str, enabled: bool):
        """Enable or disable the hook implementation for a specific plugin.

        Parameters
        ----------
        plugin_name : str
            The name of a plugin implementing ``hook_spec``.
        enabled : bool
            Whether or not the implementation should be enabled.

        Raises
        ------
        KeyError
            If ``plugin_name`` has not provided a hook implementation for this
            hook specification.
        """
        self.get_hookimpl_for_plugin(plugin_name).enabled = enabled

    def enable_plugin(self, plugin_name: str):
        """enable implementation for ``plugin_name``."""
        self._set_plugin_enabled(plugin_name, True)

    def disable_plugin(self, plugin_name: str):
        """disable implementation for ``plugin_name``."""
        self._set_plugin_enabled(plugin_name, False)


pluggy.manager._HookCaller = _HookCaller


class PluginManager(pluggy.PluginManager):
    PLUGIN_ENTRYPOINT = "napari.plugin"
    PLUGIN_PREFIX = "napari_"

    def __init__(
        self,
        project_name: str = "napari",
        autodiscover: Optional[Union[bool, str]] = True,
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

        # project_name might not be napari if running tests
        if project_name == 'napari':
            # define hook specifications and validators
            self.add_hookspecs(hook_specifications)

            # register our own built plugins
            self.register(_builtins, name='builtins')

            # discover external plugins
            if not os.environ.get("NAPARI_DISABLE_PLUGIN_AUTOLOAD"):
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
        if path:
            sys.path.insert(0, path)

        count = 0
        for plugin_name, module_name in iter_plugin_modules(
            prefix=self.PLUGIN_PREFIX, group=self.PLUGIN_ENTRYPOINT
        ):
            if self.get_plugin(plugin_name) or self.is_blocked(plugin_name):
                continue
            try:
                self._register_module(plugin_name, module_name)
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

    def _register_module(self, plugin_name: str, module_name: str):
        """Try to register `module_name` as a plugin named `plugin_name`.

        Parameters
        ----------
        plugin_name : str
            The name given to the plugin in the plugin manager.
        module_name : str
            The importable module name

        Raises
        ------
        PluginImportError
            If an error is raised when trying to import `module_name`
        PluginRegistrationError
            If an error is raised when trying to register the plugin (such as
            a PluginValidationError.)
        """
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
) -> Generator[importlib_metadata.EntryPoint, None, None]:
    """Yield all entry_points matching "group", from any distribution.

    Distribution here refers more specifically to the information in the
    dist-info folder that usually accompanies an installed package.  If a
    package in the environment does *not* have a ``dist-info/entry_points.txt``
    file, then in will not be discovered by this function.

    Note: a single package may provide multiple entrypoints for a given group.

    Parameters
    ----------
    group : str
        The name of the entry point to search.

    Yields
    -------
    Generator[importlib_metadata.EntryPoint, None, None]
        [description]

    Example
    -------
    >>> list(entry_points_for('napari.plugin'))
    [EntryPoint(name='napari-reg', value='napari_reg', group='napari.plugin'),
     EntryPoint(name='myplug', value='another.module', group='napari.plugin')]

    """
    for dist in importlib_metadata.distributions():
        for ep in dist.entry_points:
            if ep.group == group:
                yield ep


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


# regex to parse importlib_metadata.EntryPoint.value strings
# entry point format:  "name = module.with.periods:attr [extras]"
entry_point_pattern = re.compile(
    r'(?P<module>[\w.]+)\s*'
    r'(:\s*(?P<attr>[\w.]+))?\s*'
    r'(?P<extras>\[.*\])?\s*$'
)


def iter_plugin_modules(
    prefix: Optional[str] = None, group: Optional[str] = None
) -> Generator[Tuple[str, str], None, None]:
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
    setup.py entry would register two plugins under the names
    "plugin_package.register" and "plugin_package.segment"

    setup(
        name="napari-plugin-package",
        entry_points={
            "napari.plugin": [
                "plugin_package.register = napari_plugin_package.registration",
                "plugin_package.segment = napari_plugin_package.segmentation"
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
        (plugin_name, module_name)
    """
    seen_modules = set()
    if group and not os.environ.get("NAPARI_DISABLE_ENTRYPOINT_PLUGINS"):
        for ep in entry_points_for(group):
            match = entry_point_pattern.match(ep.value)
            if match:
                module = match.group('module')
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
