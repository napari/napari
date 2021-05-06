from ._plugin_manager import NapariPluginManager

__all__ = ["plugin_manager", "menu_item_template"]


# the main plugin manager instance for the `napari` plugin namespace.
plugin_manager = NapariPluginManager()
plugin_manager._initialize()

#: Template to use for namespacing a plugin item in the menu bar
menu_item_template = '{}: {}'
