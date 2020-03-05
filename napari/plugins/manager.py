import importlib
import os
import pkgutil
import re
import sys
from collections import defaultdict
from logging import getLogger
from traceback import format_exception
from typing import DefaultDict, Dict, Generator, List, Optional, Tuple, Union

import pluggy

from . import _builtins, hook_specifications

logger = getLogger(__name__)

if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    import importlib_metadata


class PluginError(Exception):
    def __init__(
        self, message: str, plugin_name: str, plugin_module: str
    ) -> None:
        super().__init__(message)
        self.plugin_name = plugin_name
        self.plugin_module = plugin_module


class PluginImportError(PluginError, ImportError):
    """Raised when a plugin fails to import."""

    def __init__(self, plugin_name: str, plugin_module: str) -> None:
        msg = f'Failed to import plugin: "{plugin_name}"'
        super().__init__(msg, plugin_name, plugin_module)


class PluginRegistrationError(PluginError):
    """Raised when a plugin fails to register with pluggy."""

    def __init__(self, plugin_name: str, plugin_module: str) -> None:
        msg = f'Failed to register plugin: "{plugin_name}"'
        super().__init__(msg, plugin_name, plugin_module)


class NapariPluginManager(pluggy.PluginManager):
    PLUGIN_ENTRYPOINT = "napari.plugin"
    PLUGIN_PREFIX = "napari_"

    def __init__(
        self, autodiscover: Optional[Union[bool, str]] = True
    ) -> None:
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
        # this dict is a mapping of plugin_name: List[raised_exception_objects]
        # this will be used to retrieve traceback info on demand
        self._exceptions: DefaultDict[str, List[PluginError]] = defaultdict(
            list
        )

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
                self._exceptions[plugin_name].append(exc)
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

        if path:
            sys.path.remove(path)

        return count

    def _register_module(self, plugin_name: str, module_name: str) -> None:
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

    def format_exceptions(self, plugin_name: str) -> str:
        """Return formatted tracebacks for all exceptions raised by plugin.

        Parameters
        ----------
        plugin_name : str
            The name of a plugin for which to retrieve tracebacks

        Returns
        -------
        str
            A formatted string with traceback information for every exception
            raised by ``plugin_name`` during this session.
        """
        from napari import __version__

        if not self._exceptions.get(plugin_name):
            return ''

        _linewidth = 80
        _pad = (_linewidth - len(plugin_name) - 18) // 2
        msg = [
            f'{"=" * _pad} Errors for plugin "{plugin_name}" {"=" * _pad}',
            '',
            f'{"napari version": >16}: {__version__}',
        ]
        try:
            err0 = self._exceptions.get(plugin_name)[0]
            package_meta = fetch_module_metadata(err0.plugin_module)
            msg.extend(
                [
                    f'{"plugin name": >16}: {package_meta["name"]}',
                    f'{"version": >16}: {package_meta["version"]}',
                    f'{"module": >16}: {err0.plugin_module}',
                ]
            )
        except Exception:
            pass
        msg += ['']

        for n, err in enumerate(self._exceptions.get(plugin_name, [])):
            _pad = _linewidth - len(str(err)) - 10
            msg += ['', f'ERROR #{n + 1}:  {str(err)} {"-" * _pad}', '']
            msg.extend(format_exception(err.__class__, err, err.__traceback__))

        msg += ['=' * 80]

        return "\n".join(msg)


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


def fetch_module_metadata(distname: str) -> Optional[Dict[str, str]]:
    """Attempt to retrieve name, version, contact email & url for a package.

    Parameters
    ----------
    distname : str
        Name of a distribution.  Note: this must match the *name* of the
        package in the METADATA file... not the name of the module.

    Returns
    -------
    package_info : dict or None
        A dict with keys 'name', 'version', 'email', and 'url'.
        Returns None of the distname cannot be found.
    """
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


def log_plugin_error(exc: PluginError) -> None:
    """Log PluginError to logger, with helpful contact info if possible.

    Parameters
    ----------
    exc : PluginError
        An instance of a PluginError
    """
    from napari import __version__

    msg = f'\nPluginError: {exc}'
    if exc.__cause__:
        cause = str(exc.__cause__).replace("\n", "\n" + " " * 13)
        msg += f'\n  Cause was: {cause}'
    contact = fetch_module_metadata(exc.plugin_module)
    if contact:
        msg += "\n  Please notify the plugin developer:\n"
        extra = [f'{k: >11}: {v}' for k, v in contact.items()]
        extra += [f'{"napari": >11}: v{__version__}']
        msg += "\n".join(extra)
    logger.error(msg)
