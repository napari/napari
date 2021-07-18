"""Utility functions related to themes."""


def add_plugins_theme(self, event=None):
    """Add theme from plugins."""
    from ..utils.theme import register_theme
    from . import plugin_manager

    for theme_name, theme_data in plugin_manager.iter_themes():
        register_theme(theme_name, theme_data)
