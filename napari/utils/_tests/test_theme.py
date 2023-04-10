import os
import sys

import pytest
from npe2 import PluginManager, PluginManifest
from npe2 import __version__ as npe2_version
from npe2.manifest.schema import ContributionPoints
from packaging.version import parse as parse_version
from pydantic import ValidationError

from napari.resources._icons import PLUGIN_FILE_NAME
from napari.settings import get_settings
from napari.utils.theme import (
    Theme,
    _install_npe2_themes,
    available_themes,
    get_theme,
    is_theme_available,
    register_theme,
    unregister_theme,
)


def test_default_themes():
    themes = available_themes()
    assert 'dark' in themes
    assert 'light' in themes
    assert 'system' in themes


def test_get_theme():
    # get theme in the old-style dict format
    theme = get_theme("dark", True)
    assert isinstance(theme, dict)

    # get theme in the new model-based format
    theme = get_theme("dark", False)
    assert isinstance(theme, Theme)


def test_get_system_theme(monkeypatch):
    monkeypatch.setattr('napari.utils.theme.get_system_theme', lambda: 'light')
    theme = get_theme('system', as_dict=False)
    # should return the theme specified by get_system_theme
    assert theme.id == 'light'


def test_register_theme():
    # Check that blue theme is not listed in available themes
    themes = available_themes()
    assert 'test_blue' not in themes

    # Create new blue theme based on dark theme
    blue_theme = get_theme('dark', True)
    blue_theme.update(
        background='rgb(28, 31, 48)',
        foreground='rgb(45, 52, 71)',
        primary='rgb(80, 88, 108)',
        current='rgb(184, 112, 0)',
    )

    # Register blue theme
    register_theme('test_blue', blue_theme, "test")

    # Check that blue theme is listed in available themes
    themes = available_themes()
    assert 'test_blue' in themes

    # Check that the dark theme has not been overwritten
    dark_theme = get_theme('dark', True)
    assert dark_theme['background'] != blue_theme['background']

    # Check that blue theme can be gotten from available themes
    theme = get_theme('test_blue', True)
    assert theme['background'] == blue_theme['background']

    theme = get_theme("test_blue", False)
    assert theme.background.as_rgb() == blue_theme["background"]


def test_unregister_theme():
    # Create new blue theme based on dark theme
    blue_theme = get_theme('dark', True)
    blue_theme.update(
        background='rgb(28, 31, 48)',
        foreground='rgb(45, 52, 71)',
        primary='rgb(80, 88, 108)',
        current='rgb(184, 112, 0)',
    )

    # Register blue theme
    register_theme('test_blue', blue_theme, "test")

    # Check that blue theme is listed in available themes
    themes = available_themes()
    assert 'test_blue' in themes

    # Remove theme from available themes
    unregister_theme("test_blue")
    themes = available_themes()
    assert 'test_blue' not in themes


def test_rebuild_theme_settings():
    settings = get_settings()
    assert "another-theme" not in available_themes()
    # theme is not updated
    with pytest.raises(ValidationError):
        settings.appearance.theme = "another-theme"
    blue_theme = get_theme("dark", True)
    register_theme("another-theme", blue_theme, "test")
    settings.appearance.theme = "another-theme"


@pytest.mark.skipif(
    os.getenv('CI') and sys.version_info < (3, 9),
    reason="Testing theme on CI is extremely slow ~ 15s per test."
    "Skip for now until we find the reason",
)
@pytest.mark.parametrize(
    "color",
    [
        "#FF0000",
        "white",
        (0, 127, 127),
        (0, 255, 255, 0.5),
        [50, 200, 200],
        [140, 140, 140, 0.7],
    ],
)
def test_theme(color):
    theme = get_theme("dark", False)
    theme.background = color


def test_theme_syntax_highlight():
    theme = get_theme("dark", False)
    with pytest.raises(ValidationError):
        theme.syntax_style = "invalid"


def test_is_theme_available(tmp_path, monkeypatch):
    (tmp_path / "test_blue").mkdir()
    (tmp_path / "yellow").mkdir()
    (tmp_path / "test_blue" / PLUGIN_FILE_NAME).write_text("test-blue")
    monkeypatch.setattr(
        "napari.utils.theme._theme_path", lambda x: tmp_path / x
    )

    n_themes = len(available_themes())

    def mock_install_theme(_themes):
        theme_dict = _themes["dark"].dict()
        theme_dict["id"] = "test_blue"
        register_theme("test_blue", theme_dict, "test")

    monkeypatch.setattr(
        "napari.utils.theme._install_npe2_themes", mock_install_theme
    )

    assert len(available_themes()) == n_themes
    assert is_theme_available("dark")
    assert not is_theme_available("green")
    assert not is_theme_available("yellow")
    assert is_theme_available("test_blue")
    assert len(available_themes()) == n_themes + 1
    assert "test_blue" in available_themes()


@pytest.mark.skipif(
    parse_version(npe2_version) < parse_version("0.6.2"),
    reason="requires npe2 0.6.2 for syntax style contributions",
)
def test_theme_registration(monkeypatch, caplog):
    data = {"dark": get_theme("dark", as_dict=False)}

    manifest = PluginManifest(
        name="theme_test",
        display_name="Theme Test",
        contributions=ContributionPoints(
            themes=[
                {
                    "id": "theme1",
                    "label": "Theme 1",
                    "type": "dark",
                    "syntax_style": "native",
                    "colors": {},
                },
                {
                    "id": "theme2",
                    "label": "Theme 2",
                    "type": "dark",
                    "syntax_style": "does_not_exist",
                    "colors": {},
                },
            ]
        ),
    )

    def mock_iter_manifests(disabled):
        return [manifest]

    monkeypatch.setattr(
        PluginManager.instance(), "iter_manifests", mock_iter_manifests
    )
    monkeypatch.setattr("napari.utils.theme._themes", data)
    _install_npe2_themes(data)

    assert "theme1" in data
    assert "theme2" not in data
    assert "Registration theme failed" in caplog.text
