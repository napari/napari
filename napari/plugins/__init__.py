from ._discovery import (
    available_samples,
    discover_dock_widgets,
    discover_sample_data,
    dock_widgets,
    function_widgets,
    get_plugin_widget,
    register_dock_widget,
)
from ._plugin_manager import plugin_manager

__all__ = [
    'available_samples',
    'discover_dock_widgets',
    'dock_widgets',
    'function_widgets',
    'get_plugin_widget',
    'menu_item_template',
    'register_dock_widget',
    "plugin_manager",
]


discover_sample_data()

#: Template to use for namespacing a plugin item in the menu bar
menu_item_template = '{}: {}'