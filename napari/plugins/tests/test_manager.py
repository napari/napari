import importlib
import os
import sys

import pytest

from napari.plugins import NapariPluginManager

# from napari.plugins.manager import load_modules_by_prefix, validate_hookimpls

PATH = os.path.join(os.path.dirname(__file__), 'fixtures')


def setup_module():
    """adding plugin test fixtures to path for autodiscovery"""
    sys.path.append(PATH)


def teardown_module():
    sys.path.pop(sys.path.index(PATH))


@pytest.fixture
def pm():
    pm = NapariPluginManager()
    return pm


def test_naming_convention_discovery(pm):
    """make sure loading by naming convention works, and doesn't crash on
    invalid plugins.

    Note: loading by setuptools entry_point is tested in pluggy and is not
    tested here.
    """
    assert 'napari_test_plugin' in pm._name2plugin
    assert 'napari_bad_plugin' in pm._name2plugin

    # napari_invalid_plugin has an invalid hookimpl, and will not get loaded
    assert 'napari_invalid_plugin' not in pm._name2plugin


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
