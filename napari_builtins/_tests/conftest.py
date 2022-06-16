from pathlib import Path
from unittest.mock import patch

import pytest
from npe2 import DynamicPlugin, PluginManager, PluginManifest

import napari_builtins


@pytest.fixture(autouse=True)
def _mock_npe2_pm():
    """Mock plugin manager with no registered plugins."""
    with patch.object(PluginManager, 'discover'):
        _pm = PluginManager()
    with patch('npe2.PluginManager.instance', return_value=_pm):
        yield _pm


@pytest.fixture(autouse=True)
def _use_builtins(_mock_npe2_pm: PluginManager):

    plugin = DynamicPlugin('napari', plugin_manager=_mock_npe2_pm)
    mf = PluginManifest.from_file(
        Path(napari_builtins.__file__).parent / 'builtins.yaml'
    )
    plugin.manifest = mf
    with plugin:
        yield plugin
