"""Tests for the settings manager.
"""

import pydantic
import pytest

from napari.utils.settings._manager import CORE_SETTINGS, SettingsManager
from napari.utils.theme import get_theme, register_theme


@pytest.fixture
def settings(tmp_path):
    return SettingsManager(tmp_path, save_to_disk=True)


def test_settings_file(tmp_path):
    SettingsManager(tmp_path, save_to_disk=True)
    fpath = tmp_path / "settings.yaml"
    assert fpath.exists()


def test_settings_file_not_created(tmp_path):
    SettingsManager(tmp_path, save_to_disk=False)
    fpath = tmp_path / "settings.yaml"
    assert not fpath.exists()


def test_settings_get_section_name():
    class SomeSectionSettings:
        pass

    section = SettingsManager._get_section_name(SomeSectionSettings)
    assert section == "somesection"


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
