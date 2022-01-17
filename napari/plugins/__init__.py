from npe2 import PluginManager as _PluginManager

from ..settings import get_settings
from ._plugin_manager import NapariPluginManager

__all__ = ["plugin_manager", "menu_item_template"]

_npe2pm = _PluginManager.instance()

# the main plugin manager instance for the `napari` plugin namespace.
plugin_manager = NapariPluginManager()
plugin_manager._initialize()

# Disable plugins listed as disabled in settings, or detected in npe2
_from_npe2 = {m.package_metadata.name for m in _npe2pm._manifests.values()}
_toblock = get_settings().plugins.disabled_plugins.union(_from_npe2)
plugin_manager._blocked.update(_toblock)

#: Template to use for namespacing a plugin item in the menu bar
menu_item_template = '{}: {}'
