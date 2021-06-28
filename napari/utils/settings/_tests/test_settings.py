"""Tests for the settings manager.
"""

from pathlib import Path

import pydantic
import pytest

from napari.utils import settings as settings_module
from napari.utils.settings import NapariSettings, _SettingsProxy
from napari.utils.theme import get_theme, register_theme


@pytest.fixture
def settings(tmp_path):
    class TestSettings(NapariSettings):
        class Config:
            env_prefix = 'testnapari_'

    return TestSettings(tmp_path / 'test_settings.yml')


def test_settings_file(settings):
    assert not Path(settings.config_path).exists()
    settings.save()
    assert Path(settings.config_path).exists()


def test_settings_autosave(settings):
    assert not Path(settings.config_path).exists()
    settings.appearance.theme = 'light'
    assert Path(settings.config_path).exists()


def test_settings_file_not_created(settings):
    assert not Path(settings.config_path).exists()
    settings._save_on_change = False
    settings.appearance.theme = 'light'
    assert not Path(settings.config_path).exists()


def test_settings_loads(tmp_path):
    data = "appearance:\n   theme: light"
    fake_path = tmp_path / 'fake_path.yml'
    fake_path.write_text(data)
    assert NapariSettings(fake_path).appearance.theme == "light"


def test_settings_load_invalid_content(tmp_path):
    # This is invalid content

    fake_path = tmp_path / 'fake_path.yml'
    fake_path.write_text(":")
    with pytest.warns(UserWarning):
        NapariSettings(fake_path)


# def test_settings_load_invalid_type():
#     # The invalid data will be replaced by the default value
#     settings = NapariSettings(application={'first_time': [1, 2]})
#     assert settings.application.first_time is True


# def test_settings_init_invalid_key():
#     # The invalid key will be removed
#     settings = NapariSettings(application={'non_existing_key': True})
#     assert getattr(settings, "non_existing_key") is None


# def test_settings_load_invalid_key(tmp_path):
#     # The invalid key will be removed
#     fake_path = tmp_path / 'fake_path.yml'
#     fake_path.write_text("application:\n\tnon_existing_key: [1, 2]")

#     settings = NapariSettings(tmp_path)
#     print(settings)
#     # assert getattr(settings, "non_existing_key") is None


# def test_settings_load_invalid_section(tmp_path):
#     # The invalid section will be removed from the file
#     data = "non_existing_section:\n   foo: bar"

#     fake_path = tmp_path / 'fake_path.yml'
#     fake_path.write_text(data)

#     settings = NapariSettings(fake_path)
#     assert getattr(settings, "non_existing_section") is None


def test_settings_to_dict(settings):
    data_dict = settings.dict()
    assert isinstance(data_dict, dict) and data_dict.get("application")

    data_dict = settings.dict(exclude_defaults=True)
    assert not data_dict.get("application")


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


# def test_settings_string(settings):
#     assert 'application:\n' in str(settings)


def test_model_fields_are_annotated(settings):
    errors = []
    for field in settings.__fields__.values():
        model = field.type_
        difference = set(model.__fields__) - set(model.__annotations__)
        if difference:
            errors.append(
                f"Model '{model.__name__}' does not provide annotations "
                f"for the fields:\n{', '.join(repr(f) for f in difference)}"
            )

    if errors:
        raise ValueError("\n\n".join(errors))


def test_settings_env_variables(monkeypatch):
    assert NapariSettings().appearance.theme == 'dark'
    # NOTE: this was previously tested as NAPARI_THEME
    monkeypatch.setenv('NAPARI_APPEARANCE_THEME', 'light')
    assert NapariSettings().appearance.theme == 'light'

    # can also use json
    assert NapariSettings().application.first_time is True
    # NOTE: this was previously tested as NAPARI_THEME
    monkeypatch.setenv('NAPARI_APPLICATION', '{"first_time": "false"}')
    assert NapariSettings().application.first_time is False


def test_settings_env_variables_fails(monkeypatch):
    monkeypatch.setenv('NAPARI_APPEARANCE_THEME', 'FOOBAR')
    with pytest.raises(pydantic.ValidationError):
        NapariSettings()


# # Failing because dark is actually the default...
# def test_settings_env_variables_do_not_write_to_disk(tmp_path, monkeypatch):

#     data = "appearance:\n   theme: light"
#     fake_path = tmp_path / 'fake_path.yml'
#     fake_path.write_text(data)

#     disk_settings = fake_path.read_text()
#     assert 'theme: light' in disk_settings
#     assert NapariSettings(fake_path).appearance.theme == "light"

#     monkeypatch.setenv('NAPARI_APPEARANCE_THEME', 'dark')
#     settings = NapariSettings(fake_path)
#     assert settings.appearance.theme == 'dark'
#     settings.save()

#     disk_settings = fake_path.read_text()
#     assert 'theme: light' in disk_settings
#     assert NapariSettings(fake_path).appearance.theme == "light"


def test_settings_only_saves_non_default_values(tmp_path):
    from yaml import safe_load

    # manually get all default data and write to yaml file
    all_data = NapariSettings(None).yaml()
    fake_path = tmp_path / 'fake_path.yml'
    assert 'appearance' in all_data
    assert 'application' in all_data
    fake_path.write_text(all_data)

    # load that yaml file and resave
    NapariSettings(fake_path).save()

    # make sure that it's now just an empty dict
    assert not safe_load(fake_path.read_text())


def test_get_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(settings_module, "SETTINGS", _SettingsProxy())
    settings = settings_module.get_settings(tmp_path)
    assert settings._config_path == tmp_path


def test_get_settings_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(settings_module, "SETTINGS", _SettingsProxy())
    settings_module.get_settings(tmp_path)
    with pytest.raises(Exception) as e:
        settings_module.get_settings(tmp_path)

    assert 'The path can only be set once per session' in str(e)
