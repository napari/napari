import importlib
import os
import sys

import pytest

from napari.plugins import NapariPluginManager


@pytest.fixture
def pm():
    fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures')
    pm = NapariPluginManager(autodiscover=fixture_path)
    assert fixture_path not in sys.path, 'discover path leaked into sys.path'
    return pm


def test_naming_convention_discovery(pm):
    """make sure loading by naming convention works, and doesn't crash on
    invalid plugins.
    """
    assert 'napari_test_plugin' in pm._name2plugin
    assert 'napari_bad_plugin' in pm._name2plugin

    # napari_invalid_plugin has an invalid hookimpl, and will not get loaded
    assert 'napari_invalid_plugin' not in pm._name2plugin


def test_entry_points_discovery(pm):
    """make sure loading by entry_point works, and doesn't crash on invalid
    plugins.
    """
    assert 'working' in pm._name2plugin
    # invalid_plugin has an invalid hookimpl, and will not get loaded
    assert 'invalid' not in pm._name2plugin
    # unimportable raises an exception during import... shouldn't make it.
    assert 'unimportable' not in pm._name2plugin


def test_invalid_plugin_raises(pm):
    """Plugins that break the hookspec API will raise PluginValidationError."""
    with pytest.raises(NapariPluginManager.PluginValidationError):
        bad = importlib.import_module('napari_invalid_plugin')
        pm.register(bad)


def test_disable_autodiscover_with_env_var():
    """Test that plugin discovery can be disabled with env var"""
    os.environ["NAPARI_DISABLE_PLUGIN_AUTOLOAD"] = '1'
    pm = NapariPluginManager()
    assert 'napari_test_plugin' not in pm._name2plugin
    del os.environ["NAPARI_DISABLE_PLUGIN_AUTOLOAD"]

    os.environ["NAPARI_DISABLE_NAMEPREFIX_PLUGINS"] = '1'
    pm = NapariPluginManager()
    assert 'napari_test_plugin' not in pm._name2plugin
    del os.environ["NAPARI_DISABLE_NAMEPREFIX_PLUGINS"]
