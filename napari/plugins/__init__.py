import sys
import os

from napari_plugin_engine import PluginManager
from . import hook_specifications
from . import _builtins

if os.name == 'nt':
    # This is where plugins will be in bundled apps on windows
    exe_dir = os.path.dirname(sys.executable)
    winlib = os.path.join(exe_dir, "Lib", "site-packages")
    sys.path.append(winlib)

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
