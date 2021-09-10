"""Test `napari_experimental_provide_icons` hook specification."""
from typing import TYPE_CHECKING

import pytest
from napari_plugin_engine import napari_hook_implementation

from napari.resources import ICONS

if TYPE_CHECKING:
    from napari.plugins._plugin_manager import NapariPluginManager


def _get_icons():
    return list(ICONS.values())[0:3]


def test_provide_icons_hook(napari_plugin_manager: "NapariPluginManager"):
    class TestPlugin:
        @napari_hook_implementation
        def napari_experimental_provide_icons():
            return _get_icons()

    napari_plugin_manager.discover_icons()
    napari_plugin_manager.register(TestPlugin)

    # make sure theme data is present in the plugin
    reg = napari_plugin_manager._icons_data["TestPlugin"]
    assert isinstance(reg, dict)
    assert len(reg) == 3  # ensure only 3 icons were added

    for icon in reg.keys():
        assert icon.startswith("TestPlugin:")
        assert icon in ICONS


def test_provide_icons_hook_bad(napari_plugin_manager: "NapariPluginManager"):
    class TestPluginBad:
        @napari_hook_implementation
        def napari_experimental_provide_icons():
            return ["NOT_A_PATH", "NOT_A_PATH.svg"]

    napari_plugin_manager.discover_icons()
    with pytest.warns(
        UserWarning,
        match="Plugin 'TestPluginBad' provided icon",
    ):
        napari_plugin_manager.register(TestPluginBad)

    # make sure theme data is present in the plugin
    reg = napari_plugin_manager._icons_data["TestPluginBad"]
    assert isinstance(reg, dict)
    assert len(reg) == 0  # ensure no icons were added


def test_provide_icons_hook_bad_not_iterable(
    napari_plugin_manager: "NapariPluginManager",
):
    class TestPluginBad:
        @napari_hook_implementation
        def napari_experimental_provide_icons():
            return {"A": "A", "B": "B"}

    napari_plugin_manager.discover_icons()
    with pytest.warns(
        UserWarning,
        match="Plugin 'TestPluginBad' provided a non-iterable object",
    ):
        napari_plugin_manager.register(TestPluginBad)

    # make sure theme data is present in the plugin
    assert "TestPluginBad" not in napari_plugin_manager._icons_data


def test_provide_icons_hook_unregister(
    napari_plugin_manager: "NapariPluginManager",
):
    class TestPlugin:
        @napari_hook_implementation
        def napari_experimental_provide_icons():
            return _get_icons()

    napari_plugin_manager.discover_icons()
    napari_plugin_manager.register(TestPlugin)

    # make sure theme data is present in the plugin
    reg = napari_plugin_manager._icons_data["TestPlugin"]
    assert isinstance(reg, dict)
    assert len(reg) == 3  # ensure only 3 icons were added
    for icon in reg.keys():
        assert icon in ICONS

    # unregister icons
    napari_plugin_manager.unregister("TestPlugin")
    assert "TestPlugin" not in napari_plugin_manager._icons_data
    for icon in reg.keys():
        assert icon not in ICONS
