"""Tests for the settings manager.
"""

import pydantic
import pytest
from yaml import safe_load

from napari.utils.settings._manager import CORE_SETTINGS, SettingsManager
from napari.utils.theme import get_theme, register_theme


@pytest.fixture
def settings(tmp_path):
    return SettingsManager(tmp_path, save_to_disk=True)


def test_settings_file(tmp_path):
    SettingsManager(tmp_path, save_to_disk=True)
    fpath = tmp_path / SettingsManager._FILENAME
    assert fpath.exists()


def test_settings_file_not_created(tmp_path):
    SettingsManager(tmp_path, save_to_disk=False)
    fpath = tmp_path / SettingsManager._FILENAME
    assert not fpath.exists()


def test_settings_loads(tmp_path):
    data = """
appearance:
  theme: light
"""
    with open(tmp_path / SettingsManager._FILENAME, "w") as fh:
        fh.write(data)

    settings = SettingsManager(tmp_path)
    assert settings.appearance.theme == "light"


def test_settings_load_invalid_type(tmp_path):
    # The invalid data will be replaced by the default value
    data = """
application:
  first_time: [1, 2]
"""
    with open(tmp_path / SettingsManager._FILENAME, "w") as fh:
        fh.write(data)

    settings = SettingsManager(tmp_path)
    assert settings.application.first_time is True


def test_settings_load_invalid_key(tmp_path):
    # The invalid key will be removed
    data = """
application:
  non_existing_key: [1, 2]
"""
    with open(tmp_path / SettingsManager._FILENAME, "w") as fh:
        fh.write(data)

    settings = SettingsManager(tmp_path)
    assert getattr(settings, "non_existing_key") is None


def test_settings_load_invalid_section(tmp_path):
    # The invalid section will be removed from the file
    data = """
non_existing_section:
  foo: bar
"""
    with open(tmp_path / SettingsManager._FILENAME, "w") as fh:
        fh.write(data)

    settings = SettingsManager(tmp_path)
    assert getattr(settings, "non_existing_section") is None


def test_settings_to_dict(settings):
    data_dict = settings._to_dict()
    assert "application" in data_dict
    assert isinstance(data_dict, dict)


def test_settings_reset(settings):
    settings.reset()
    assert settings.appearance.theme == "dark"
    settings.appearance.theme = "light"
    assert settings.appearance.theme == "light"
    settings.reset()
    assert settings.appearance.theme == "dark"


def test_settings_schemas(settings):
    for _, data in settings.schemas().items():
        assert "json_schema" in data
        assert "model" in data


def test_settings_model(settings):
    with pytest.raises(pydantic.error_wrappers.ValidationError):
        # Should be string
        settings.appearance.theme = 1

    with pytest.raises(pydantic.error_wrappers.ValidationError):
        # Should be a valid string
        settings.appearance.theme = "vaporwave"


def test_custom_theme_settings(settings):
    # See: https://github.com/napari/napari/issues/2340
    custom_theme_name = "_test_blue_"

    # No theme registered yet, this should fail
    with pytest.raises(pydantic.error_wrappers.ValidationError):
        settings.appearance.theme = custom_theme_name

    blue_theme = get_theme('dark')
    blue_theme.update(
        background='rgb(28, 31, 48)',
        foreground='rgb(45, 52, 71)',
        primary='rgb(80, 88, 108)',
        current='rgb(184, 112, 0)',
    )
    register_theme(custom_theme_name, custom_theme_name)

    # Theme registered, should pass validation
    settings.appearance.theme = custom_theme_name


def test_settings_string(settings):
    assert 'application:\n' in str(settings)


def test_settings_load_invalid_content(tmp_path):
    # This is invalid content
    data = ":"
    with open(tmp_path / SettingsManager._FILENAME, "w") as fh:
        fh.write(data)

    with pytest.warns(UserWarning):
        SettingsManager(tmp_path)


def test_model_fields_are_annotated():
    errors = []
    for model in CORE_SETTINGS:
        difference = set(model.__fields__) - set(model.__annotations__)
        if difference:
            errors.append(
                f"Model '{model.__name__}' does not provide annotations "
                f"for the fields:\n{', '.join(repr(f) for f in difference)}"
            )

    if errors:
        raise ValueError("\n\n".join(errors))


def test_settings_env_variables(tmp_path, monkeypatch):
    value = 'light'
    monkeypatch.setenv('NAPARI_THEME', value)
    settings = SettingsManager(tmp_path, save_to_disk=True)
    assert CORE_SETTINGS[0]().theme == value
    assert settings.appearance.theme == value


def test_settings_env_variables_do_not_write_to_disk(tmp_path, monkeypatch):
    data = """
appearance:
  theme: pink
"""
    with open(tmp_path / SettingsManager._FILENAME, "w") as fh:
        fh.write(data)

    value = 'light'
    monkeypatch.setenv('NAPARI_THEME', value)
    settings = SettingsManager(tmp_path, save_to_disk=True)
    settings._save()

    with open(tmp_path / SettingsManager._FILENAME) as fh:
        saved_data = fh.read()

    model_values = settings._remove_default(settings._to_dict(safe=True))
    saved_values = safe_load(saved_data)

    assert model_values["appearance"]["theme"] == value
    # Note: Pink is currently not a valid theme, but if we use dark as it is the
    # default it is not saved in the saved_values. We can't use "Light" either
    assert saved_values["appearance"]["theme"] == "pink"

    model_values["appearance"].pop("theme")
    saved_values["appearance"].pop("theme")
    assert model_values == saved_values


def test_settings_env_variables_fails(tmp_path, monkeypatch):
    value = 'FOOBAR'
    monkeypatch.setenv('NAPARI_THEME', value)
    with pytest.raises(pydantic.error_wrappers.ValidationError):
        SettingsManager(tmp_path, save_to_disk=True)


def test_core_settings_are_class_variables_in_settings_manager():
    for setting in CORE_SETTINGS:
        schema = setting.schema()
        section = schema["section"]
        assert section in SettingsManager.__annotations__
        assert setting == SettingsManager.__annotations__[section]
