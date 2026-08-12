from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from npe2._pytest_plugin import TestPluginManager
from napari.settings._plugin_config_generator import (
    _build_single_config_model,
    plugin_configuration_generator,
)

PLUGIN_NAME = 'my-plugin'  # this matches the sample_manifest
MANIFEST_PATH = (
    Path(__file__).parent.parent.parent
    / 'plugins/_tests/_sample_manifest.yaml'
)


@pytest.fixture
def mock_pm(npe2pm: 'TestPluginManager'):
    from napari.plugins import _initialize_plugins

    _initialize_plugins.cache_clear()
    mock_reg = MagicMock()
    npe2pm._command_registry = mock_reg
    with npe2pm.tmp_plugin(manifest=MANIFEST_PATH):
        yield npe2pm


def test_single_config(mock_pm: 'TestPluginManager'):
    configs = mock_pm.get_manifest(PLUGIN_NAME).contributions.configurations
    assert len(configs) == 2
    assert configs['reader'].title == 'Reading with something'
    assert configs['reader'].properties['lazy'].type == 'boolean'

    model = _build_single_config_model(configs['writer'], 'writer')
    # generated field names are the manifest property keys, verbatim
    assert set(model.model_fields) == set(configs['writer'].properties)

    plugin_prefs = plugin_configuration_generator(mock_pm)['my-plugin']
    # the model class name is a valid identifier even when the plugin name
    # (a PEP-508 package name like 'my-plugin') is not
    assert plugin_prefs.__name__.isidentifier()
    str(plugin_prefs)
