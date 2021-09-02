from napari.utils.theme import available_themes, get_theme, register_theme


def test_default_themes():
    themes = available_themes()
    assert 'dark' in themes
    assert 'light' in themes


def test_get_theme():
    theme = get_theme('dark')
    assert theme['folder'] == 'dark'


def test_register_theme():
    # Check that blue theme is not listed in available themes
    themes = available_themes()
    assert 'test_blue' not in themes

    # Create new blue theme based on dark theme
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
    dark_theme = get_theme('dark')
    assert not dark_theme['background'] == blue_theme['background']

    # Check that blue theme can be gotten from available themes
    theme = get_theme('test_blue')
    assert theme['background'] == blue_theme['background']
