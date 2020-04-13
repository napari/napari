import sys

from pluggy import HookimplMarker

from .exceptions import PluginError, PluginImportError, PluginRegistrationError
from .manager import get_plugin_manager, PluginManager

# Marker to be imported and used in plugins (and for own implementations)
# Note: plugins may also just import pluggy directly and make their own
# napari_hook_implementation.
napari_hook_implementation = HookimplMarker("napari")


__all__ = [
    "napari_hook_implementation",
    "get_plugin_manager",
    "PluginManager",
    "PluginError",
    "PluginImportError",
    "PluginRegistrationError",
]
