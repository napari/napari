"""Test `napari_experimental_provide_qss` hook specification."""
from typing import TYPE_CHECKING

import pytest
from napari_plugin_engine import napari_hook_implementation

from napari._qt.qt_resources import STYLES

if TYPE_CHECKING:
    from napari.plugins._plugin_manager import NapariPluginManager


@pytest.fixture
def get_qss(tmp_path):
    """Fixture that provides list of qss files."""

    def _get_qss():
        d = tmp_path / "qss"
        d.mkdir()
        file = d / "04_file.qss"
        file.write_text("")
        return [file]

    return _get_qss


def test_provide_qss_hook(
    napari_plugin_manager: "NapariPluginManager", get_qss
):
    class TestPlugin:
        @napari_hook_implementation
        def napari_experimental_provide_qss():
            return get_qss()

    napari_plugin_manager.discover_qss()
    napari_plugin_manager.register(TestPlugin)

    # make sure theme data is present in the plugin
    reg = napari_plugin_manager._qss_data["TestPlugin"]
    assert isinstance(reg, dict)
    assert len(reg) == 1  # ensure only 1 file was added

    for qss in reg.keys():
        assert qss.startswith("TestPlugin:")
        assert qss in STYLES


def test_provide_qss_hook_bad(napari_plugin_manager: "NapariPluginManager"):
    class TestPluginBad:
        @napari_hook_implementation
        def napari_experimental_provide_qss():
            return ["NOT_A_PATH", "NOT_A_PATH.qss"]

    napari_plugin_manager.discover_qss()
    with pytest.warns(
        UserWarning,
        match="Plugin 'TestPluginBad' provided stylesheet",
    ):
        napari_plugin_manager.register(TestPluginBad)

    # make sure theme data is present in the plugin
    reg = napari_plugin_manager._qss_data["TestPluginBad"]
    assert isinstance(reg, dict)
    assert len(reg) == 0  # ensure no stylesheets were added


def test_provide_qss_hook_bad_not_iterable(
    napari_plugin_manager: "NapariPluginManager",
):
    class TestPluginBad:
        @napari_hook_implementation
        def napari_experimental_provide_qss():
            return {"A": "A", "B": "B"}

    napari_plugin_manager.discover_qss()
    with pytest.warns(
        UserWarning,
        match="Plugin 'TestPluginBad' provided a non-iterable object",
    ):
        napari_plugin_manager.register(TestPluginBad)

    # make sure theme data is present in the plugin
    assert "TestPluginBad" not in napari_plugin_manager._qss_data


def test_provide_qss_hook_unregister(
    napari_plugin_manager: "NapariPluginManager", get_qss
):
    class TestPlugin:
        @napari_hook_implementation
        def napari_experimental_provide_qss():
            return get_qss()

    napari_plugin_manager.discover_qss()
    napari_plugin_manager.register(TestPlugin)

    # make sure theme data is present in the plugin
    reg = napari_plugin_manager._qss_data["TestPlugin"]
    assert isinstance(reg, dict)
    assert len(reg) == 1  # ensure only single stylesheet was added

    # unregister stylesheets
    napari_plugin_manager.unregister("TestPlugin")
    assert "TestPlugin" not in napari_plugin_manager._qss_data

    for qss in reg.keys():
        assert qss not in STYLES
