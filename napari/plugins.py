import importlib
import pkgutil
import re
import warnings
from collections import abc
from typing import Iterator

import pkg_resources
import requests


"""
Plugin discovery follows two of the conventions here:
https://packaging.python.org/guides/creating-and-discovering-plugins/

1) Using naming convention:
    plugins installed in the environment that follow a naming convention
    (e.g. "napari_plugin"), can be discovered using `pkgutil`.
    This also enables easy discovery on pypi

2) Using package metadata:
    plugins that declare a special key (e.g. "napari.plugins") in their
    setup.py `entry_points` can be discovered using `pkg_resources`.
"""


class PluginManager(abc.Mapping):
    PLUGIN_PREFIX = 'napari_'  # for discovery using naming convention
    PLUGIN_ENTRY_POINT = "napari.plugins"  # for discovery using entry_points

    def __init__(self, autodiscover: bool = True):
        """Mapping between plugin module names and imported modules.

        Parameters
        ----------
        autodiscover : bool, optional
            Whether to autodiscover plugin modules on init, by default True
        """
        self._plugins = {}
        if autodiscover:
            self.discover()

    def __getitem__(self, item):
        return self._plugins[item]

    def __iter__(self):
        return iter(self._plugins)

    def __len__(self):
        return len(self._plugins)

    def discover(self):
        """Discover napari plugins in the environment.

        This uses two methods:
            1. naming convention: http://bit.ly/pynaming-convention
                looks for all installed packages starting with self.PREFIX
                convention for plugins is: napari_<pluginname>
            2. package metadata: http://bit.ly/pypackage-meta
                installed packages can register themselves for discovery by
                providing the `entry_points` argument in setup.py using
                self.PLUGIN_ENTRY_POINT
        """
        # discover plugins using naming convention
        self._plugins.update(
            {
                name: importlib.import_module(name)
                for finder, name, ispkg in pkgutil.iter_modules()
                if name.startswith(self.PLUGIN_PREFIX)
            }
        )
        # discover plugins using package metadata
        self._plugins.update(
            {
                entry.name: entry.load()
                for entry in pkg_resources.iter_entry_points(
                    self.PLUGIN_ENTRY_POINT
                )
            }
        )

    def register(self, name: str):
        """Manually register a module name as a plugin

        Parameters
        ----------
        name : str
            name of an importable module
        """
        self._plugins.update({name: importlib.import_module(name)})

    @property
    def readers(self) -> Iterator[tuple]:
        """Generator that yields all readers declared in plugins.

        Plugins may declare a top level variable `READERS` which is a list of
        2-tuples [(checker, reader) ...] where:
            `checker` is a function that returns True if it recognizes a
            directory as something it can handle
            `reader` is a function that accepts args (path, viewer) and
            adds layers to the viewer given a path.

        Yields
        -------
        tuple
            (name, module) for all plugins
        """
        for name, module in self.items():
            for item in getattr(module, 'READERS', []):
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and all([callable(i) for i in item])
                ):
                    yield item
                else:
                    warnings.warn(
                        'Reader {item} from plugin {name} was not loaded '
                        'because it is not a 2-tuple of callables. If you did '
                        'not write this plugin, please alert the developer.'
                    )

    def get_pypi_plugins(self) -> dict:
        """Search for napari plugins on pypi.

        Packages using naming convention: http://bit.ly/pynaming-convention
        can be autodiscovered on pypi using the SIMPLE API:
        https://www.python.org/dev/peps/pep-0503/

        Returns
        -------
        dict
            {name: url} for all modules at pypi that start with self.PREFIX
        """
        PYPI_SIMPLE_API_URL = 'https://pypi.org/simple/'

        if not hasattr(self, '_pypi_plugins'):
            response = requests.get(PYPI_SIMPLE_API_URL)
            if response.status_code == 200:
                pattern = f'<a href="/simple/(.+)">({self.PREFIX}.*)</a>'
                self._pypi_plugins = {
                    name: PYPI_SIMPLE_API_URL + url
                    for url, name in re.findall(pattern, response.text)
                }
        return self._pypi_plugins

    def fetch_plugin_versions(self, name: str) -> tuple:
        """Fetch available pypi versions for a plugin name

        Parameters
        ----------
        name : str
            name of the plugin

        Returns
        -------
        tuple of versions availabe on pypi

        Raises
        ------
        KeyError
            if the plugin name is not found on pypi
        """
        if name not in self.pypi_plugins and not name.startswith(
            self.PPLUGIN_PREFIXREFIX
        ):
            # also search for plugin preceeded by self.PREFIX
            name = f'{self.PREFIX}-{name}'
        if name not in self.pypi_plugins:
            raise KeyError(f'"{name}"" is not a recognized plugin name')

        response = requests.get(self.pypi_plugins.get(name))
        response.raise_for_status()
        versions = re.findall(f'>{name}-(.+).tar', response.text)
        return tuple(set(versions))


plugins = PluginManager()
