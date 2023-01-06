from functools import lru_cache
from app_model.types import KeyBindingRule

from npe2 import PackageMetadata
from npe2 import PluginManager as _PluginManager
from npe2 import PluginManifest

from napari.plugins import _npe2
from napari.plugins._plugin_manager import NapariPluginManager
from napari.settings import get_settings

__all__ = ("plugin_manager", "menu_item_template")

from napari.utils.theme import _install_npe2_themes

#: Template to use for namespacing a plugin item in the menu bar
# widget_name (plugin_name)
menu_item_template = '{1} ({0})'
"""Template to use for namespacing a plugin item in the menu bar"""
#: The main plugin manager instance for the `napari` plugin namespace.
plugin_manager = NapariPluginManager()
"""Main Plugin manager instance"""


@lru_cache  # only call once
def _initialize_plugins():
    _npe2pm = _PluginManager.instance()

    settings = get_settings()
    if settings.schema_version >= '0.4.0':
        for p in settings.plugins.disabled_plugins:
            _npe2pm.disable(p)

    # just in case anything has already been registered before we initialized
    _npe2.on_plugins_registered(set(_npe2pm.iter_manifests()))

    # connect enablement/registration events to listeners
    _npe2pm.events.enablement_changed.connect(
        _npe2.on_plugin_enablement_change
    )
    _npe2pm.events.plugins_registered.connect(_npe2.on_plugins_registered)
    _npe2pm.discover(include_npe1=settings.plugins.use_npe2_adaptor)

    # this is a workaround for the fact that briefcase does not seem to include
    # napari's entry_points.txt in the bundled app, so the builtin plugins
    # don't get detected.  So we just register it manually.  This could
    # potentially be removed when we move to a different bundle strategy
    if 'napari' not in _npe2pm._manifests:
        mf = PluginManifest.from_distribution('napari')
        mf.package_metadata = PackageMetadata.for_package('napari')
        _npe2pm.register(mf)

    # Disable plugins listed as disabled in settings, or detected in npe2
    _from_npe2 = {m.name for m in _npe2pm.iter_manifests()}
    if 'napari' in _from_npe2:
        _from_npe2.update({'napari', 'builtins'})
    plugin_manager._skip_packages = _from_npe2
    plugin_manager._blocked.update(settings.plugins.disabled_plugins)

    if settings.plugins.use_npe2_adaptor:
        # prevent npe1 plugin_manager discovery
        # (this doesn't prevent manual registration)
        plugin_manager.discover = lambda *a, **k: None
    else:
        plugin_manager._initialize()

    _install_npe2_themes()

    # FIXME is this the right place to initialize plugin keybindings?
    from napari.utils.shortcuts import plugins_shortcuts
    from napari._app_model import get_app

    shortcut_settings = get_settings().shortcuts.shortcuts
    for command, keybinding_rules in plugins_shortcuts.items():
        if command in shortcut_settings:
            for keybinding in shortcut_settings[command]:
                get_app().keybindings.register_keybinding_rule(command, KeyBindingRule(primary=str(keybinding)))
        else:
            for keybinding_rule in keybinding_rules:
                get_app().keybindings.register_keybinding_rule(command, keybinding_rule)
