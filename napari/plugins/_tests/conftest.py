import os
import sys

import pytest
from napari_plugin_engine import PluginManager

from napari.plugins import hook_specifications, _builtins


@pytest.fixture
def plugin_manager():
    """PluginManager fixture that loads some test plugins"""
    fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures')
    plugin_manager = PluginManager(
        'napari',
        discover_entry_point='napari.plugin',
        discover_prefix='napari_',
    )
    plugin_manager.add_hookspecs(hook_specifications)
    plugin_manager.register(_builtins, name='builtins')
    plugin_manager.discover(fixture_path)
    assert fixture_path not in sys.path, 'discover path leaked into sys.path'
    return plugin_manager


@pytest.fixture
def builtin_plugin_manager(plugin_manager):
    for mod in plugin_manager.get_plugins():
        if mod != _builtins:
            plugin_manager.unregister(mod)
    assert plugin_manager.get_plugins() == set([_builtins])
    return plugin_manager
