from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from npe2._pytest_plugin import TestPluginManager
from napari.settings._plugin_config_generator import (
    _build_single_config_model,
    _snake_identifier,
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


def test_snake_identifier():
    assert (
        _snake_identifier('demoplugin.value1.value2', 'demoplugin')
        == 'value1_value2'
    )
    assert (
        _snake_identifier('Demo Configuration for widget 1')
        == 'demo_configuration_for_widget_1'
    )
    # keys may be namespaced with the plugin name using a different separator
    # than the plugin name itself ('my_plugin.' in the key vs 'my-plugin')
    assert (
        _snake_identifier('my_plugin.reader.lazy', 'my-plugin')
        == 'reader_lazy'
    )


def test_single_config(mock_pm: 'TestPluginManager'):
    configs = mock_pm.get_manifest(PLUGIN_NAME).contributions.configuration
    assert len(configs) == 2
    assert configs[0].title == 'Demo Configuration for widget 1'
    assert configs[0].properties['my_plugin.reader.lazy'].type == 'boolean'

    model = _build_single_config_model(configs[0], PLUGIN_NAME)
    assert len(model.model_fields) == 2
    # model/field names must be valid Python identifiers so the parent
    # preferences model can be built (pydantic < 2.9 raises otherwise)
    assert all(name.isidentifier() for name in model.model_fields)
    str(plugin_configuration_generator(mock_pm)['my-plugin'])
