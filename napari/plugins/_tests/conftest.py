import pytest
import os
import sys
from napari.plugins import NapariPluginManager
import napari.plugins._builtins


@pytest.fixture
def plugin_manager():
    """PluginManager fixture that loads some test plugins"""
    fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures')
    plugin_manager = NapariPluginManager(autodiscover=fixture_path)
    assert fixture_path not in sys.path, 'discover path leaked into sys.path'
    return plugin_manager


@pytest.fixture
def builtin_plugin_manager(plugin_manager):
    for mod in plugin_manager.get_plugins():
        if mod != napari.plugins._builtins:
            plugin_manager.unregister(mod)
    assert plugin_manager.get_plugins() == set([napari.plugins._builtins])
    return plugin_manager
