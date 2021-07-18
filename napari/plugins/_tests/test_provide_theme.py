from typing import TYPE_CHECKING

import pytest
from napari_plugin_engine import napari_hook_implementation

from napari.utils.theme import available_themes, get_theme
from napari.viewer import ViewerModel

if TYPE_CHECKING:
    from napari.plugins._plugin_manager import NapariPluginManager


def test_provide_theme_hook(napari_plugin_manager: "NapariPluginManager"):

    viewer = ViewerModel()
    napari_plugin_manager.discover_themes()

    class TestPlugin:
        @napari_hook_implementation
        def napari_provide_theme():
            dark = get_theme("dark")
            dark["folder"] = "dark-test"
            return [dark]

    napari_plugin_manager.register(TestPlugin)

    # make sure theme data is present in the plugin
    reg = napari_plugin_manager._theme_data["TestPlugin"]
    assert isinstance(reg, dict)
    assert len(reg) == 1

    # make sure theme was registered
    assert "dark-test" in available_themes()
    viewer.theme = "dark-test"

    class TestPluginBad:
        @napari_hook_implementation
        def napari_provide_theme():
            dark = get_theme("dark")
            dark.pop("foreground")
            dark["folder"] = "dark-bad"
            return [dark]

    with pytest.warns(
        UserWarning,
        match=", plugin 'TestPluginBad' provided an invalid dict object",
    ):
        napari_plugin_manager.register(TestPluginBad)

    # make sure theme data is present in the plugin
    reg = napari_plugin_manager._theme_data["TestPluginBad"]
    assert isinstance(reg, dict)
    assert len(reg) == 0

    class TestPluginMissing:
        @napari_hook_implementation
        def napari_provide_theme():
            dark = get_theme("dark")
            dark.pop("folder")
            return [dark]

    with pytest.warns(
        UserWarning,
        match="KeyError",
    ):
        napari_plugin_manager.register(TestPluginMissing)

    # make sure theme data is present in the plugin
    reg = napari_plugin_manager._theme_data["TestPluginMissing"]
    assert isinstance(reg, dict)
    assert len(reg) == 0
