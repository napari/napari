import pytest

from napari.utils.theme import (
    Theme,
    available_themes,
    get_theme,
    register_theme,
    unregister_theme,
)


def test_default_themes():
    themes = available_themes()
    assert 'dark' in themes
    assert 'light' in themes


def test_get_theme():
    with pytest.warns(FutureWarning):
        theme = get_theme('dark')
    assert isinstance(theme, dict)
    assert theme['folder'] == 'dark'

    # get theme in the old-style dict format
    with pytest.warns(FutureWarning):
        theme = get_theme("dark")
        assert isinstance(theme, dict)

    # get theme in the new model-based format
    theme = get_theme("dark", False)
    assert isinstance(theme, Theme)


def test_register_theme():
    # Check that blue theme is not listed in available themes
    themes = available_themes()
    assert 'test_blue' not in themes

    # Create new blue theme based on dark theme
    with pytest.warns(FutureWarning):
        blue_theme = get_theme('dark')
    blue_theme.update(
        background='rgb(28, 31, 48)',
        foreground='rgb(45, 52, 71)',
        primary='rgb(80, 88, 108)',
        current='rgb(184, 112, 0)',
    )

    # Register blue theme
    register_theme('test_blue', blue_theme)

    # Check that blue theme is listed in available themes
    themes = available_themes()
    assert 'test_blue' in themes

    # Check that the dark theme has not been overwritten
    with pytest.warns(FutureWarning):
        dark_theme = get_theme('dark')
        assert not dark_theme['background'] == blue_theme['background']

    # Check that blue theme can be gotten from available themes
    with pytest.warns(FutureWarning):
        theme = get_theme('test_blue')
        assert theme['background'] == blue_theme['background']

    theme = get_theme("test_blue", False)
    assert theme.background.as_rgb() == blue_theme["background"]


def test_unregister_theme():
    # Create new blue theme based on dark theme
    with pytest.warns(FutureWarning):
        blue_theme = get_theme('dark')
    blue_theme.update(
        background='rgb(28, 31, 48)',
        foreground='rgb(45, 52, 71)',
        primary='rgb(80, 88, 108)',
        current='rgb(184, 112, 0)',
    )

    # Register blue theme
    register_theme('test_blue', blue_theme)

    # Check that blue theme is listed in available themes
    themes = available_themes()
    assert 'test_blue' in themes

    # Remove theme from available themes
    unregister_theme("test_blue")
    themes = available_themes()
    assert 'test_blue' not in themes
