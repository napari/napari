import importlib
import os
import sys

import pytest
from pluggy import PluginValidationError

from napari.plugins import NapariPluginManager, manager


@pytest.fixture
def pm():
    """PluginManager fixture that loads some test plugins"""
    fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures')
    pm = NapariPluginManager(autodiscover=fixture_path)
    assert fixture_path not in sys.path, 'discover path leaked into sys.path'
    return pm


def test_plugin_autodiscovery(pm):
    """make sure loading by naming convention works, and doesn't crash on
    invalid plugins.
    """
    assert 'working' in pm._name2plugin
    # note, entry_points supercede naming convention
    # the "napari_working_plugin" distribution points to "napari_test_plugin"
    # but the name of the distribution is "working"...
    # so while the "napari_test_plugin" *module* fits the naming convention
    # it will appear in the plugins using the entry_points entry instead.
    assert 'napari_test_plugin' not in pm._name2plugin

    assert 'napari_bad_plugin' in pm._name2plugin

    # napari_invalid_plugin has an invalid hook implementation, and will load
    assert 'napari_invalid_plugin' not in pm._name2plugin
    # invalid_plugin has an invalid hook implementation, and will not load
    assert 'invalid' not in pm._name2plugin
    # unimportable raises an exception during import... shouldn't make it.
    assert 'unimportable' not in pm._name2plugin


def test_invalid_plugin_raises(pm):
    """Plugins that break the hookspec API will raise PluginValidationError."""
    with pytest.raises(PluginValidationError):
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


def test_iter_plugins():
    """Test that plugin discovery is working."""
    fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures')
    sys.path.append(fixture_path)

    # Search by entry_point group only
    assert set(manager.iter_plugin_modules(group='napari.plugin')).issuperset(
        {
            ('unimportable', 'unimportable_plugin'),
            ('invalid', 'invalid_plugin'),
            ('working', 'napari_test_plugin'),
        }
    )

    # Search by name_convention only
    assert set(manager.iter_plugin_modules(prefix='napari_')).issuperset(
        {
            ('napari_bad_plugin', 'napari_bad_plugin'),
            ('napari_invalid_plugin', 'napari_invalid_plugin'),
            ('napari_test_plugin', 'napari_test_plugin'),
            ('napari_unimportable_plugin', 'napari_unimportable_plugin'),
        }
    )

    # Search by BOTH name_convention and entry_point...
    # note that the plugin name for plugin "working" is taken from the
    # entry_point, and not from the module name...
    assert set(
        manager.iter_plugin_modules(prefix='napari', group='napari.plugin')
    ).issuperset(
        {
            ('unimportable', 'unimportable_plugin'),
            ('invalid', 'invalid_plugin'),
            ('working', 'napari_test_plugin'),
            ('napari_bad_plugin', 'napari_bad_plugin'),
            ('napari_invalid_plugin', 'napari_invalid_plugin'),
            ('napari_unimportable_plugin', 'napari_unimportable_plugin'),
        }
    )

    sys.path.remove(fixture_path)
