import sys

from napari_plugins import napari_hook_implementation

from .exceptions import PluginError, PluginImportError, PluginRegistrationError
from naplugi import PluginManager
from . import hook_specifications
from . import _builtins

# the main plugin manager instance for the `napari` plugin namespace.
plugin_manager = PluginManager('napari')
plugin_manager.add_hookspecs(hook_specifications)
plugin_manager.register(_builtins, name='builtins')
plugin_manager.hook._needs_discovery = True


__all__ = [
    "PluginManager",
    "plugin_manager",
    "PluginError",
    "PluginImportError",
    "PluginRegistrationError",
    "napari_hook_implementation",
]
