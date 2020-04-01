import sys

from pluggy import HookimplMarker

from ._hook_callers import execute_hook
from .exceptions import PluginError, PluginImportError, PluginRegistrationError
from .manager import PluginManager

# Marker to be imported and used in plugins (and for own implementations)
# Note: plugins may also just import pluggy directly and make their own
# napari_hook_implementation.
napari_hook_implementation = HookimplMarker("napari")

# the main plugin manager instance for the `napari` plugin namespace.
plugin_manager = PluginManager()

__all__ = [
    "napari_hook_implementation",
    "PluginManager",
    "plugin_manager",
    "PluginError",
    "PluginImportError",
    "PluginRegistrationError",
    "execute_hook",
]
