import importlib
import pkgutil
import re

import pkg_resources
import requests

"""
This proposal for plugin discovery follows two of the recommendations here:
https://packaging.python.org/guides/creating-and-discovering-plugins/

1) Using naming convention:
    plugins installed in the environment that follow a naming convention
    (e.g. "napari_fancy_plugin"), can be discovered using `pkgutil`.
    This also enables easy discovery on pypi

2) Using package metadata:
    plugins that declare a special key (e.g. "napari.plugins") in their
    setup.py `entry_points` can be discovered using `pkg_resources`.
    (this is used by pytest, for example)
"""


class PluginManager:
    PREFIX = 'napari_'  # for discovery using naming convention
    PLUGIN_ENTRY_POINT = "napari.plugins"  # for discovery using entry_points
    IO_PLUGINS = "napari_io_"  # plugin subtype... this couple be an enum?
    PYPI_SIMPLE_API_URL = 'https://pypi.org/simple/'

    def __init__(self, discover=True):
        self.plugins = {}
        if discover:
            self.discover_plugins()

    @property
    def readers(self):
        """only plugins that declare themselves as io plugins.

        the super basic API example I'm using here is that io plugins may
        declare:
            plugin.READERS : a list of 2-tuples (checker, reader)
                `checker` is a function that returns True if it recognizes a
                directory as something it can handle
                `reader` is a function that accepts args (path, viewer) and
                adds layers to the viewer given a path.
            plugin.WRITERS : not implemented...

        Yields
        -------
        tuple
            (name, module) for all plugins
        """
        for name, module in self.plugins.items():
            if name.startswith(self.IO_PLUGINS):
                for item in getattr(module, 'READERS', []):
                    yield item

    def discover_plugins(self):
        # using naming convention: http://bit.ly/pynaming-convention
        # looks for all installed packages starting with self.PREFIX
        # propsed convention for plugins is: napari_<plugin-name>
        # coupld potentially have subclasses like napari_io_readerplugin
        self.plugins.update(
            {
                name: importlib.import_module(name)
                for finder, name, ispkg in pkgutil.iter_modules()
                if name.startswith(self.PREFIX)
            }
        )

        # using package metadata: http://bit.ly/pypackage-meta
        # installed packages can register themselves for discovery by providing
        # the `entry_points` argument in setup.py using self.PLUGIN_ENTRY_POINT
        self.plugins.update(
            {
                entry.name: entry.load()
                for entry in pkg_resources.iter_entry_points(
                    self.PLUGIN_ENTRY_POINT
                )
            }
        )

    @property
    def pypi_plugins(self):
        # packages using naming convention: http://bit.ly/pynaming-convention
        # can be autodiscovered on pypi using the SIMPLE API:
        # https://www.python.org/dev/peps/pep-0503/
        if not hasattr(self, '_pypi_plugins'):
            response = requests.get(self.PYPI_SIMPLE_API_URL)
            if response.status_code == 200:
                pattern = f'<a href="/simple/(.+)">({self.PREFIX}.*)</a>'
                self._pypi_plugins = {
                    name: self.PYPI_SIMPLE_API_URL + url
                    for url, name in re.findall(pattern, response.text)
                }
        return self._pypi_plugins

    def fetch_plugin_versions(self, name):
        """fetch versions for a plugin.

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
        if name not in self.pypi_plugins and not name.startswith(self.PREFIX):
            # also search for plugin preceeded by self.PREFIX
            name = f'{self.PREFIX}-{name}'
        if name not in self.pypi_plugins:
            raise KeyError(f'"{name}"" is not a recognized plugin name')

        response = requests.get(self.pypi_plugins.get(name))
        if response.status_code == 200:
            versions = re.findall(f'>{name}-(.+).tar', response.text)
            return tuple(set(versions))


plugin_manager = PluginManager()
