from unittest.mock import patch

import pytest
from napari_plugin_engine import napari_hook_implementation

from napari._qt import Window
from napari.utils.theme import Theme, get_theme


@patch.object(Window, "_remove_theme")
@patch.object(Window, "_add_theme")
def test_provide_theme_hook_registered_correctly(
    mock_add_theme,
    mock_remove_theme,
    make_napari_viewer,
    napari_plugin_manager,
):
    dark = get_theme("dark", True)
    dark["name"] = "dark-test-2"

    class TestPlugin:
        @napari_hook_implementation
        def napari_experimental_provide_theme():
            return {"dark-test-2": dark}

    # create instance of viewer to make sure
    # registration and unregistration methods are called
    viewer = make_napari_viewer()

    # register theme
    napari_plugin_manager.register(TestPlugin)
    reg = napari_plugin_manager._theme_data["TestPlugin"]
    assert isinstance(reg["dark-test-2"], Theme)

    viewer.theme = "dark-test-2"

    # triggered when theme was added
    mock_add_theme.assert_called()
    mock_remove_theme.assert_not_called()

    # now, lets unregister the theme
    with pytest.warns(UserWarning, match="The current theme "):
        napari_plugin_manager.unregister("TestPlugin")
    mock_remove_theme.assert_called()
