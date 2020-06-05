import sys
import os

from napari_plugin_engine import PluginManager as _PM
from . import hook_specifications
from . import _builtins


# move to napari-plugin-engine
class PluginManager(_PM):
    def prune(self):
        for plugin in list(self.plugins):
            meta = self.get_standard_metadata(plugin)
            meta.pop("hooks")
            meta.pop("plugin_name")
            meta.pop("version")
            if not any(i for i in meta.values()):
                self.unregister(plugin)


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
