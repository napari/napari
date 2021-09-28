from typing import Any

from ..settings import get_settings
from ._plugin_manager import NapariPluginManager

__all__ = ["plugin_manager", "menu_item_template"]


class _NapariPluginManagerProxy:
    """Backwards compatibility layer."""

    def __getattribute__(self, name) -> Any:
        return getattr(get_plugin_manager(), name)


plugin_manager = _NapariPluginManagerProxy()
_plugin_manager = None


def get_plugin_manager():
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = NapariPluginManager()
        _plugin_manager._initialize()
        # Disable plugins listed as disabled in settings.
        _plugin_manager._blocked.update(
            get_settings().plugins.disabled_plugins
        )
    return _plugin_manager


#: Template to use for namespacing a plugin item in the menu bar
menu_item_template = '{}: {}'
