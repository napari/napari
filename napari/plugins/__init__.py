import sys

from napari_plugins import napari_hook_implementation

from .exceptions import PluginError, PluginImportError, PluginRegistrationError
from naplugi import PluginManager
from . import hook_specifications
from . import _builtins

# the main plugin manager instance for the `napari` plugin namespace.
plugin_manager = PluginManager(
    'napari', discover_entrypoint='napari.plugin', discover_prefix='napari_'
)
with plugin_manager.discovery_blocked():
    plugin_manager.add_hookspecs(hook_specifications)
    plugin_manager.register(_builtins, name='builtins')


__all__ = [
    "PluginManager",
    "plugin_manager",
    "PluginError",
    "PluginImportError",
    "PluginRegistrationError",
    "napari_hook_implementation",
]
