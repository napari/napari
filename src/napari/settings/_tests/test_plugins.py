from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from npe2._pytest_plugin import TestPluginManager
from napari.settings._plugin_config_generator import (
    _build_single_config_model,
    _field_name,
    plugin_configuration_generator,
)

PLUGIN_NAME = 'my-plugin'  # this matches the sample_manifest
PLUGIN_DISPLAY_NAME = 'My Plugin'  # this matches the sample_manifest
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


def test_field_name():
    plugin_name = 'demoplugin'
    key = 'demoplugin.value1.value2'

    assert _field_name(key, plugin_name) == 'value1_value2'


def test_single_config(mock_pm: 'TestPluginManager'):
    configs = mock_pm.get_manifest(PLUGIN_NAME).contributions.configuration
    assert len(configs) == 2
    assert configs[0].title == 'Demo Configuration for widget 1'
    assert configs[0].properties['my_plugin.reader.lazy'].type == 'boolean'

    assert (
        len(_build_single_config_model(configs[0], PLUGIN_NAME).model_fields)
        == 2
    )
    assert plugin_configuration_generator(mock_pm)
