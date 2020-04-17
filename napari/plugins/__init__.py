import sys

from .exceptions import PluginError, PluginImportError, PluginRegistrationError
from .manager import PluginManager

# the main plugin manager instance for the `napari` plugin namespace.
plugin_manager = PluginManager()

__all__ = [
    "PluginManager",
    "plugin_manager",
    "PluginError",
    "PluginImportError",
    "PluginRegistrationError",
]
