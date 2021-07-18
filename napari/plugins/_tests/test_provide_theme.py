from typing import TYPE_CHECKING

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

    # make sure theme was registered
    assert "dark-test" in available_themes()
    viewer.theme = "dark-test"
