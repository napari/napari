"""Tests for the settings manager."""

from pathlib import Path

import pydantic
import pytest

from napari.utils.settings import NapariSettings
from napari.utils.theme import get_theme, register_theme


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


def test_settings_to_dict(test_settings):
    data_dict = test_settings.dict()
    assert isinstance(data_dict, dict) and data_dict.get("application")

    data_dict = test_settings.dict(exclude_defaults=True)
    assert not data_dict.get("application")


def test_settings_reset(test_settings):
    test_settings.reset()
    assert test_settings.appearance.theme == "dark"
    test_settings.appearance.theme = "light"
    assert test_settings.appearance.theme == "light"
    test_settings.reset()
    assert test_settings.appearance.theme == "dark"


def test_settings_schemas(test_settings):
    for _, data in test_settings.schemas().items():
        assert "json_schema" in data
        assert "model" in data


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

    blue_theme = get_theme('dark')
    blue_theme.update(
        background='rgb(28, 31, 48)',
        foreground='rgb(45, 52, 71)',
        primary='rgb(80, 88, 108)',
        current='rgb(184, 112, 0)',
    )
    register_theme(custom_theme_name, custom_theme_name)

    # Theme registered, should pass validation
    test_settings.appearance.theme = custom_theme_name


# def test_settings_string(settings):
#     assert 'application:\n' in str(settings)


def test_model_fields_are_annotated(test_settings):
    errors = []
    for field in test_settings.__fields__.values():
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
    assert NapariSettings(None).appearance.theme == 'dark'
    # NOTE: this was previously tested as NAPARI_THEME
    monkeypatch.setenv('NAPARI_APPEARANCE_THEME', 'light')
    assert NapariSettings(None).appearance.theme == 'light'

    # can also use json
    assert NapariSettings(None).application.first_time is True
    # NOTE: this was previously tested as NAPARI_THEME
    monkeypatch.setenv('NAPARI_APPLICATION', '{"first_time": "false"}')
    assert NapariSettings(None).application.first_time is False


def test_settings_env_variables_fails(monkeypatch):
    monkeypatch.setenv('NAPARI_APPEARANCE_THEME', 'FOOBAR')
    with pytest.raises(pydantic.ValidationError):
        NapariSettings()


def test_subfield_env_field(monkeypatch):
    """test that setting Field(env=) works for subfields"""
    from napari.utils.settings._base import EventedSettings

    class Sub(EventedSettings):
        x: int = pydantic.Field(1, env='varname')

    class T(NapariSettings):
        sub: Sub

    monkeypatch.setenv("VARNAME", '42')
    assert T().sub.x == 42


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


def test_get_settings(tmp_path):
    from napari.utils import settings

    s = settings.get_settings(tmp_path)
    assert s.config_path == tmp_path


def test_get_settings_fails(monkeypatch, tmp_path):
    from napari.utils import settings

    settings.get_settings(tmp_path)
    with pytest.raises(Exception) as e:
        settings.get_settings(tmp_path)

    assert 'The path can only be set once per session' in str(e)


def test_first_time():
    """This test just confirms that we don't load an existing file (locally)"""
    assert NapariSettings().application.first_time is True
