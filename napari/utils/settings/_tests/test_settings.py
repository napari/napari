"""Tests for the settings manager.
"""

import pydantic
import pytest

from napari.utils.settings._manager import SettingsManager


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
        class Config:
            title = "Some Section Settings"

    section = SettingsManager._get_section_name(SomeSectionSettings())
    assert section == "some_section"


def test_settings_loads(tmp_path):
    data = """
application:
  theme: light
"""
    with open(tmp_path / SettingsManager._FILENAME, "w") as fh:
        fh.write(data)

    settings = SettingsManager(tmp_path)
    assert settings.application.theme == "light"


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
    assert settings.application.theme == "dark"
    settings.application.theme = "light"
    assert settings.application.theme == "light"
    settings.reset()
    assert settings.application.theme == "dark"


def test_settings_schemas(settings):
    for _, data in settings.schemas().items():
        assert "json_schema" in data
        assert "model" in data


def test_settings_model(settings):
    with pytest.raises(pydantic.error_wrappers.ValidationError):
        # Should be string
        settings.application.theme = 1

    with pytest.raises(pydantic.error_wrappers.ValidationError):
        # Should be a valid string
        settings.application.theme = "vaporwave"
