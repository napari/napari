"""Tests for the ``plugin_settings`` fixture provided by napari's pytest plugin.

The ``plugin_settings`` fixture (in ``napari.utils._testsupport``) makes
``napari.settings.get_plugin_settings`` hermetic per-test: it clears the
``_PLUGIN_PREFERENCES`` cache, points the default ``path_dir`` at the test's
``tmp_path``, and connects ``_clear_plugin_settings_cache`` to npe2's
``plugins_registered`` / ``enablement_changed`` signals so plugins registered
mid-test are picked up.
"""

from npe2 import PluginManifest

PLUGIN_A = PluginManifest(
    name='plugin-a',
    display_name='Plugin A',
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
                    'max_size_mb': {
                        'type': 'integer',
                        'default': 512,
                        'minimum': 0,
                        'maximum': 4096,
                        'title': 'Max size (MB)',
                    },
                    'colormap': {
                        'type': 'string',
                        'default': 'gray',
                        'title': 'Colormap',
                        'enum': ['gray', 'green', 'viridis'],
                    },
                },
            },
        }
    },
)

PLUGIN_B = PluginManifest(
    name='plugin-b',
    display_name='Plugin B',
    contributions={
        'configurations': {
            'writer': {
                'title': 'Writer settings',
                'properties': {
                    'compress': {
                        'type': 'boolean',
                        'default': True,
                        'title': 'Compress',
                    },
                },
            },
        }
    },
)


def test_fixture_end_to_end(plugin_settings, npe2pm):
    from napari.settings import get_plugin_settings

    npe2pm.register(PLUGIN_A)

    s = get_plugin_settings('plugin-a')
    assert s.reader.max_size_mb == 512  # manifest default
    assert s.reader.lazy is False
    assert s.config_path.name == 'plugin-a.yaml'
    # settings auto-save under the test's tmp_path, not the real config dir
    assert 'plugin-a.yaml' in str(s.config_path)

    s.reader.max_size_mb = 1024
    assert 'max_size_mb: 1024' in s.config_path.read_text()


def test_enum_property(plugin_settings, npe2pm):
    from napari.settings import get_plugin_settings

    npe2pm.register(PLUGIN_A)
    s = get_plugin_settings('plugin-a')

    assert s.reader.colormap == 'gray'  # manifest default
    # `enum` is settings-UI metadata: it stays in the model's JSON schema
    # (the Preferences widgets read it from there) but must not be passed to
    # pydantic `Field()` as an (invalid) kwarg.
    schema = type(s.reader).model_json_schema()
    assert schema['properties']['colormap']['enum'] == [
        'gray',
        'green',
        'viridis',
    ]


def test_dynamic_registration_invalidates_cache(plugin_settings, npe2pm):
    from napari.settings import get_plugin_settings

    npe2pm.register(PLUGIN_A)
    get_plugin_settings()  # build cache with only plugin-a

    # registering a new plugin should invalidate the cache so it is rebuilt
    npe2pm.register(PLUGIN_B)
    s = get_plugin_settings('plugin-b')
    assert s.writer.compress is True
    # and plugin-a is still there
    assert 'plugin-a' in get_plugin_settings()


def test_register_accepts_string_path(plugin_settings, npe2pm, tmp_path):
    from napari.settings import get_plugin_settings

    # a hand-written manifest, as a real plugin would ship it
    manifest_path = tmp_path / 'napari.yaml'
    manifest_path.write_text(
        """
name: plugin-c
display_name: Plugin C
contributions:
  configurations:
    reader:
      title: Reader settings
      properties:
        lazy:
          type: boolean
          default: false
          title: Load lazily
"""
    )
    npe2pm.register(str(manifest_path))
    assert get_plugin_settings('plugin-c').reader.lazy is False
