"""Tests for the settings manager."""
import os
from pathlib import Path

import pydantic
import pytest
from yaml import safe_load

from napari import settings
from napari.settings import CURRENT_SCHEMA_VERSION, NapariSettings
from napari.utils.theme import get_theme, register_theme


@pytest.fixture
def test_settings(tmp_path):
    """A fixture that can be used to test and save settings"""
    from napari.settings import NapariSettings

    class TestSettings(NapariSettings):
        class Config:
            env_prefix = 'testnapari_'

    return TestSettings(
        tmp_path / 'test_settings.yml', schema_version=CURRENT_SCHEMA_VERSION
    )


def test_settings_file(test_settings):
    assert not Path(test_settings.config_path).exists()
    test_settings.save()
    assert Path(test_settings.config_path).exists()


def test_settings_autosave(test_settings):
    assert not Path(test_settings.config_path).exists()
    test_settings.appearance.theme = 'light'
    assert Path(test_settings.config_path).exists()


def test_settings_file_not_created(test_settings):
    assert not Path(test_settings.config_path).exists()
    test_settings._save_on_change = False
    test_settings.appearance.theme = 'light'
    assert not Path(test_settings.config_path).exists()


def test_settings_loads(tmp_path):
    data = "appearance:\n   theme: light"
    fake_path = tmp_path / 'fake_path.yml'
    fake_path.write_text(data)
    assert NapariSettings(fake_path).appearance.theme == "light"


def test_settings_load_invalid_content(tmp_path):
    # This is invalid content

    fake_path = tmp_path / 'fake_path.yml'
    fake_path.write_text(":")
    NapariSettings(fake_path)


def test_settings_load_invalid_type(tmp_path, caplog):
    # The invalid data will be replaced by the default value
    data = "appearance:\n   theme: 1"
    fake_path = tmp_path / 'fake_path.yml'
    fake_path.write_text(data)
    assert NapariSettings(fake_path).application.save_window_geometry is True
    assert 'Validation errors in config file' in str(caplog.records[0])


def test_settings_load_strict(tmp_path, monkeypatch):
    # use Config.strict_config_check to enforce good config files
    monkeypatch.setattr(NapariSettings.__config__, 'strict_config_check', True)
    data = "appearance:\n   theme: 1"
    fake_path = tmp_path / 'fake_path.yml'
    fake_path.write_text(data)
    with pytest.raises(pydantic.ValidationError):
        NapariSettings(fake_path)


def test_settings_load_invalid_key(tmp_path, monkeypatch):
    # The invalid key will be removed

    fake_path = tmp_path / 'fake_path.yml'
    data = """
    application:
       non_existing_key: [1, 2]
       first_time: false
    """
    fake_path.write_text(data)

    monkeypatch.setattr(os, 'environ', {})
    s = NapariSettings(fake_path)
    assert getattr(s, "non_existing_key", None) is None
    s.save()
    text = fake_path.read_text()
    # removed bad key
    assert safe_load(text) == {
        'application': {'first_time': False},
        'schema_version': CURRENT_SCHEMA_VERSION,
    }


def test_settings_load_invalid_section(tmp_path):
    # The invalid section will be removed from the file
    data = "non_existing_section:\n   foo: bar"

    fake_path = tmp_path / 'fake_path.yml'
    fake_path.write_text(data)

    settings = NapariSettings(fake_path)
    assert getattr(settings, "non_existing_section", None) is None


def test_settings_to_dict(test_settings):
    data_dict = test_settings.dict()
    assert isinstance(data_dict, dict) and data_dict.get("application")

    data_dict = test_settings.dict(exclude_defaults=True)
    assert not data_dict.get("application")


def test_settings_to_dict_no_env(monkeypatch):
    """Test that exclude_env works to exclude variables coming from the env."""
    s = NapariSettings(None, appearance={'theme': 'light'})
    assert s.dict()['appearance']['theme'] == 'light'
    assert s.dict(exclude_env=True)['appearance']['theme'] == 'light'

    monkeypatch.setenv("NAPARI_APPEARANCE_THEME", 'light')
    s = NapariSettings(None)
    assert s.dict()['appearance']['theme'] == 'light'
    assert 'theme' not in s.dict(exclude_env=True).get('appearance', {})


def test_settings_reset(test_settings):
    appearance_id = id(test_settings.appearance)
    test_settings.reset()
    assert id(test_settings.appearance) == appearance_id
    assert test_settings.appearance.theme == "dark"
    test_settings.appearance.theme = "light"
    assert test_settings.appearance.theme == "light"
    test_settings.reset()
    assert test_settings.appearance.theme == "dark"
    assert id(test_settings.appearance) == appearance_id


def test_settings_model(test_settings):
    with pytest.raises(pydantic.error_wrappers.ValidationError):
        # Should be string
        test_settings.appearance.theme = 1

    with pytest.raises(pydantic.error_wrappers.ValidationError):
        # Should be a valid string
        test_settings.appearance.theme = "vaporwave"


def test_custom_theme_settings(test_settings):
    # See: https://github.com/napari/napari/issues/2340
    custom_theme_name = "_test_blue_"

    # No theme registered yet, this should fail
    with pytest.raises(pydantic.error_wrappers.ValidationError):
        test_settings.appearance.theme = custom_theme_name

    blue_theme = get_theme('dark', True)
    blue_theme.update(
        background='rgb(28, 31, 48)',
        foreground='rgb(45, 52, 71)',
        primary='rgb(80, 88, 108)',
        current='rgb(184, 112, 0)',
    )
    register_theme(custom_theme_name, blue_theme, "test")

    # Theme registered, should pass validation
    test_settings.appearance.theme = custom_theme_name


def test_settings_string(test_settings):
    setstring = str(test_settings)
    assert 'NapariSettings (defaults excluded)' in setstring
    assert 'appearance:' not in setstring
    assert repr(test_settings) == setstring


def test_model_fields_are_annotated(test_settings):
    errors = []
    for field in test_settings.__fields__.values():
        model = field.type_
        if not hasattr(model, '__fields__'):
            continue
        difference = set(model.__fields__) - set(model.__annotations__)
        if difference:
            errors.append(
                f"Model '{model.__name__}' does not provide annotations "
                f"for the fields:\n{', '.join(repr(f) for f in difference)}"
            )

    if errors:
        raise ValueError("\n\n".join(errors))


def test_settings_env_variables(monkeypatch):
    assert NapariSettings(None).appearance.theme == 'dark'
    # NOTE: this was previously tested as NAPARI_THEME
    monkeypatch.setenv('NAPARI_APPEARANCE_THEME', 'light')
    assert NapariSettings(None).appearance.theme == 'light'

    # can also use json
    assert NapariSettings(None).application.first_time is True
    # NOTE: this was previously tested as NAPARI_THEME
    monkeypatch.setenv('NAPARI_APPLICATION', '{"first_time": "false"}')
    assert NapariSettings(None).application.first_time is False

    # can also use json in nested vars
    assert NapariSettings(None).plugins.extension2reader == {}
    monkeypatch.setenv('NAPARI_PLUGINS_EXTENSION2READER', '{"*.zarr": "hi"}')
    assert NapariSettings(None).plugins.extension2reader == {"*.zarr": "hi"}


def test_settings_env_variables_fails(monkeypatch):
    monkeypatch.setenv('NAPARI_APPEARANCE_THEME', 'FOOBAR')
    with pytest.raises(pydantic.ValidationError):
        NapariSettings()


def test_subfield_env_field(monkeypatch):
    """test that setting Field(env=) works for subfields"""
    from napari.settings._base import EventedSettings

    class Sub(EventedSettings):
        x: int = pydantic.Field(1, env='varname')

    class T(NapariSettings):
        sub: Sub

    monkeypatch.setenv("VARNAME", '42')
    assert T(sub={}).sub.x == 42


# Failing because dark is actually the default...
def test_settings_env_variables_do_not_write_to_disk(tmp_path, monkeypatch):
    # create a settings file with light theme
    data = "appearance:\n   theme: light"
    fake_path = tmp_path / 'fake_path.yml'
    fake_path.write_text(data)

    # make sure they wrote correctly
    disk_settings = fake_path.read_text()
    assert 'theme: light' in disk_settings
    # make sure they load correctly
    assert NapariSettings(fake_path).appearance.theme == "light"

    # now load settings again with an Env-var override
    monkeypatch.setenv('NAPARI_APPEARANCE_THEME', 'dark')
    settings = NapariSettings(fake_path)
    # make sure the override worked, and save again
    assert settings.appearance.theme == 'dark'
    # data from the config file is still "known"
    assert settings._config_file_settings['appearance']['theme'] == 'light'
    # but we know what came from env vars as well:
    assert settings.env_settings()['appearance']['theme'] == 'dark'

    # when we save it shouldn't use environment variables and it shouldn't
    # have overriden our non-default value of `theme: light`
    settings.save()
    disk_settings = fake_path.read_text()
    assert 'theme: light' in disk_settings

    # and it's back if we reread without the env var override
    monkeypatch.delenv('NAPARI_APPEARANCE_THEME')
    assert NapariSettings(fake_path).appearance.theme == "light"


def test_settings_only_saves_non_default_values(monkeypatch, tmp_path):
    from yaml import safe_load

    # prevent error during NAPARI_ASYNC tests
    monkeypatch.setattr(os, 'environ', {})

    # manually get all default data and write to yaml file
    all_data = NapariSettings(None).yaml()
    fake_path = tmp_path / 'fake_path.yml'
    assert 'appearance' in all_data
    assert 'application' in all_data
    fake_path.write_text(all_data)

    # load that yaml file and resave
    NapariSettings(fake_path).save()

    # make sure that the only value is now the schema version
    assert safe_load(fake_path.read_text()) == {
        'schema_version': CURRENT_SCHEMA_VERSION
    }


def test_get_settings(tmp_path):
    p = f'{tmp_path}.yaml'
    s = settings.get_settings(p)
    assert str(s.config_path) == str(p)


def test_get_settings_fails(monkeypatch, tmp_path):
    p = f'{tmp_path}.yaml'
    settings.get_settings(p)
    with pytest.raises(Exception) as e:
        settings.get_settings(p)

    assert 'The path can only be set once per session' in str(e)


def test_first_time():
    """This test just confirms that we don't load an existing file (locally)"""
    assert NapariSettings().application.first_time is True


# def test_deprecated_SETTINGS():
#     """Test that direct access of SETTINGS warns."""
#     from napari.settings import SETTINGS

#     with pytest.warns(FutureWarning):
#         assert SETTINGS.appearance.theme == 'dark'


def test_no_save_path():
    """trying to save without a config path is an error"""
    s = NapariSettings(config_path=None)
    assert s.config_path is None

    with pytest.raises(ValueError):
        # the original `save()` method is patched in conftest.fresh_settings
        # so we "unmock" it here to assert the failure
        NapariSettings.__original_save__(s)  # type: ignore


def test_settings_events(test_settings):
    """Test that NapariSettings emits dotted keys."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    test_settings.events.changed.connect(mock)
    test_settings.appearance.theme = 'light'

    assert mock.called
    event = mock.call_args_list[0][0][0]
    assert event.key == 'appearance.theme'
    assert event.value == 'light'

    mock.reset_mock()
    test_settings.appearance.theme = 'light'
    mock.assert_not_called()


@pytest.mark.parametrize('ext', ['yml', 'yaml', 'json'])
def test_full_serialize(test_settings: NapariSettings, tmp_path, ext):
    """Make sure that every object in the settings is serializeable.

    Should work with both json and yaml.
    """
    test_settings.save(tmp_path / f't.{ext}', exclude_defaults=False)
