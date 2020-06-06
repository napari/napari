import sys
import os

from napari_plugin_engine import PluginManager as _PM
from . import hook_specifications
from . import _builtins
from ..utils.misc import running_as_bundled_app
from ..utils._appdirs import user_site_packages


class PluginManager(_PM):
    def discover(self):
        return super().discover(user_site_packages())


# the main plugin manager instance for the `napari` plugin namespace.
plugin_manager = PluginManager(
    'napari', discover_entry_point='napari.plugin', discover_prefix='napari_'
)
with plugin_manager.discovery_blocked():
    plugin_manager.add_hookspecs(hook_specifications)
    plugin_manager.register(_builtins, name='builtins')


__all__ = [
    "PluginManager",
    "plugin_manager",
]
