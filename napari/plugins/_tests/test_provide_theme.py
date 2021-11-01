"""Test `napari_experimental_provide_theme` hook specification."""
from typing import TYPE_CHECKING

import pytest
from napari_plugin_engine import napari_hook_implementation

from napari.settings import get_settings
from napari.utils.theme import Theme, available_themes, get_theme
from napari.viewer import ViewerModel

if TYPE_CHECKING:
    from napari.plugins._plugin_manager import NapariPluginManager


def test_provide_theme_hook(napari_plugin_manager: "NapariPluginManager"):
    with pytest.warns(FutureWarning):
        dark = get_theme("dark")
        dark["name"] = "dark-test"

    class TestPlugin:
        @napari_hook_implementation
        def napari_experimental_provide_theme():
            return {"dark-test": dark}

    viewer = ViewerModel()
    napari_plugin_manager.discover_themes()
    napari_plugin_manager.register(TestPlugin)

    # make sure theme data is present in the plugin
    reg = napari_plugin_manager._theme_data["TestPlugin"]
    assert isinstance(reg, dict)
    assert len(reg) == 1
    assert isinstance(reg["dark-test"], Theme)

    # make sure theme was registered
    assert "dark-test" in available_themes()
    viewer.theme = "dark-test"


def test_provide_theme_hook_bad(napari_plugin_manager: "NapariPluginManager"):
    napari_plugin_manager.discover_themes()

    with pytest.warns(FutureWarning):
        dark = get_theme("dark")
        dark.pop("foreground")
        dark["name"] = "dark-bad"

    class TestPluginBad:
        @napari_hook_implementation
        def napari_experimental_provide_theme():
            return {"dark-bad": dark}

    with pytest.warns(
        UserWarning,
        match=", plugin 'TestPluginBad' provided an invalid dict object",
    ):
        napari_plugin_manager.register(TestPluginBad)

    # make sure theme data is present in the plugin but the theme is not there
    reg = napari_plugin_manager._theme_data["TestPluginBad"]
    assert isinstance(reg, dict)
    assert len(reg) == 0
    assert "dark-bad" not in available_themes()


def test_provide_theme_hook_not_dict(
    napari_plugin_manager: "NapariPluginManager",
):
    napari_plugin_manager.discover_themes()

    class TestPluginBad:
        @napari_hook_implementation
        def napari_experimental_provide_theme():
            return ["bad-theme", []]

    with pytest.warns(
        UserWarning,
        match="Plugin 'TestPluginBad' provided a non-dict object",
    ):
        napari_plugin_manager.register(TestPluginBad)

    # make sure theme data is present in the plugin but the theme is not there
    assert "TestPluginBad" not in napari_plugin_manager._theme_data


def test_provide_theme_hook_unregister(
    napari_plugin_manager: "NapariPluginManager",
):
    with pytest.warns(FutureWarning):
        dark = get_theme("dark")
        dark["name"] = "dark-test"

    class TestPlugin:
        @napari_hook_implementation
        def napari_experimental_provide_theme():
            return {"dark-test": dark}

    napari_plugin_manager.discover_themes()
    napari_plugin_manager.register(TestPlugin)

    # make sure theme was registered
    assert "TestPlugin" in napari_plugin_manager._theme_data
    reg = napari_plugin_manager._theme_data["TestPlugin"]
    assert isinstance(reg, dict)
    assert len(reg) == 1
    assert "dark-test" in available_themes()
    get_settings().appearance.theme = "dark-test"

    with pytest.warns(UserWarning, match="The current theme "):
        napari_plugin_manager.unregister("TestPlugin")

    # make sure that plugin-specific data was removed
    assert "TestPlugin" not in napari_plugin_manager._theme_data
    # since the plugin was unregistered, the current theme cannot
    # be the theme registered by the plugin
    assert get_settings().appearance.theme != "dark-test"
    assert "dark-test" not in available_themes()
