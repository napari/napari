import warnings
from unittest.mock import patch

import pytest
from napari_plugin_engine import napari_hook_implementation

from napari import Viewer
from napari._qt import Window
from napari._tests.utils import skip_on_win_ci
from napari.settings import get_settings
from napari.utils.theme import Theme, get_theme


@skip_on_win_ci
@patch.object(Window, "_remove_theme")
@patch.object(Window, "_add_theme")
def test_provide_theme_hook_registered_correctly(
    mock_add_theme,
    mock_remove_theme,
    make_napari_viewer,
    napari_plugin_manager,
):
    # make a viewer with a plugin & theme registered
    viewer = make_napari_viewer_with_plugin_theme(
        make_napari_viewer,
        napari_plugin_manager,
        theme_type='dark',
        name='dark-test-2',
    )

    # set the viewer theme to the plugin theme
    viewer.theme = "dark-test-2"

    # triggered when theme was added
    mock_add_theme.assert_called()
    mock_remove_theme.assert_not_called()

    # now, lets unregister the theme
    # We didn't set the setting, so ensure that no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        napari_plugin_manager.unregister("TestPlugin")
    mock_remove_theme.assert_called()


@patch.object(Window, "_remove_theme")
@patch.object(Window, "_add_theme")
def test_plugin_provide_theme_hook_set_settings_correctly(
    mock_add_theme,
    mock_remove_theme,
    make_napari_viewer,
    napari_plugin_manager,
):
    # make a viewer with a plugin & theme registered
    make_napari_viewer_with_plugin_theme(
        make_napari_viewer,
        napari_plugin_manager,
        theme_type='dark',
        name='dark-test-2',
    )
    # set the plugin theme as a setting
    get_settings().appearance.theme = "dark-test-2"

    # triggered when theme was added
    mock_add_theme.assert_called()
    mock_remove_theme.assert_not_called()

    # now, lets unregister the theme
    # We *did* set the setting, so there should be a warning
    with pytest.warns(UserWarning, match="The current theme "):
        napari_plugin_manager.unregister("TestPlugin")
    mock_remove_theme.assert_called()


def make_napari_viewer_with_plugin_theme(
    make_napari_viewer, napari_plugin_manager, *, theme_type: str, name: str
) -> Viewer:
    theme = get_theme(theme_type).to_rgb_dict()
    theme["name"] = name

    class TestPlugin:
        @napari_hook_implementation
        def napari_experimental_provide_theme():
            return {name: theme}

    # create instance of viewer to make sure
    # registration and unregistration methods are called
    viewer = make_napari_viewer()

    # register theme
    napari_plugin_manager.register(TestPlugin)
    reg = napari_plugin_manager._theme_data["TestPlugin"]
    assert isinstance(reg[name], Theme)

    return viewer
