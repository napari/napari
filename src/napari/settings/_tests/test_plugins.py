from typing import TYPE_CHECKING

from npe2 import PluginManifest

if TYPE_CHECKING:
    from npe2._pytest_plugin import TestPluginManager
from napari.settings._plugin_config_generator import (
    _build_single_config_model,
    plugin_configuration_generator,
)

PLUGIN_NAME = 'my-plugin'  # this matches the sample_manifest


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


def test_plugin_name_starting_with_digit(npe2pm: 'TestPluginManager'):
    """A valid PEP-508 plugin name may start with a digit (e.g. ``9lives``);
    the generated model class name must still be a valid Python identifier.
    """
    manifest = PluginManifest(
        name='9lives',
        display_name='9 Lives',
        contributions={
            'configurations': {
                'reader': {
                    'title': 'Reader settings',
                    'properties': {
                        'lazy': {
                            'type': 'boolean',
                            'default': False,
                            'title': 'Load lazily',
                        },
                    },
                },
            }
        },
    )
    npe2pm.register(manifest)
    plugin_prefs = plugin_configuration_generator(npe2pm)['9lives']
    assert plugin_prefs.__name__ == '_9lives'
    assert plugin_prefs.__name__.isidentifier()
