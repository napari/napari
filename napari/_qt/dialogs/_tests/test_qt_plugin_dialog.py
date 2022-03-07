import pytest
from npe2.manifest.package_metadata import PackageMetadata

from napari._qt.dialogs import qt_plugin_dialog


def _hub_or_pypi(conda_forge):
    base_data = {
        "metadata_version": "1.0",
        "version": "0.1.0",
        "summary": "some test package",
        "home_page": "http://napari.org",
        "author": "test author",
        "license": "UNKNOWN",
    }
    for i in range(2):
        yield PackageMetadata(name=f"test-name-{i}", **base_data), bool(i)


@pytest.fixture
def plugin_dialog(qtbot, monkeypatch):
    """test that QtPluginErrReporter shows any instantiated PluginErrors."""

    for method_name in ["iter_hub_plugin_info", "iter_napari_plugin_info"]:
        monkeypatch.setattr(
            qt_plugin_dialog,
            method_name,
            _hub_or_pypi,
        )

    monkeypatch.setattr(
        qt_plugin_dialog,
        "running_as_constructor_app",
        lambda: False,
    )

    widget = qt_plugin_dialog.QtPluginDialog()
    widget.show()
    qtbot.add_widget(widget)
    return widget


@pytest.fixture
def plugin_dialog_constructor(qtbot, monkeypatch):
    """test that QtPluginErrReporter shows any instantiated PluginErrors."""

    for method_name in ["iter_hub_plugin_info", "iter_napari_plugin_info"]:
        monkeypatch.setattr(
            qt_plugin_dialog,
            method_name,
            _hub_or_pypi,
        )

    monkeypatch.setattr(
        qt_plugin_dialog,
        "running_as_constructor_app",
        lambda: True,
    )
    widget = qt_plugin_dialog.QtPluginDialog()
    widget.show()
    qtbot.add_widget(widget)
    return widget


def test_filter_not_available_plugins(plugin_dialog_constructor):
    """test that filtering works."""
    item = plugin_dialog_constructor.available_list.item(0)
    widget = plugin_dialog_constructor.available_list.itemWidget(item)
    assert not widget.action_button.isEnabled()
    assert widget.warning_tooltip.isVisible()

    item = plugin_dialog_constructor.available_list.item(1)
    widget = plugin_dialog_constructor.available_list.itemWidget(item)
    assert widget.action_button.isEnabled()
    assert not widget.warning_tooltip.isVisible()


def test_filter_available_plugins(plugin_dialog):
    """test that filtering works."""
    plugin_dialog.filter("")
    assert plugin_dialog.available_list._count_visible() == 2

    plugin_dialog.filter("no-match@123")
    assert plugin_dialog.available_list._count_visible() == 0

    plugin_dialog.filter("")
    plugin_dialog.filter("test-name-0")
    assert plugin_dialog.available_list._count_visible() == 1


def test_filter_installed_plugins(plugin_dialog):
    """test that filtering works."""
    plugin_dialog.filter("")
    assert plugin_dialog.installed_list._count_visible() >= 0

    plugin_dialog.filter("no-match@123")
    assert plugin_dialog.installed_list._count_visible() == 0


def test_visible_widgets(plugin_dialog):
    assert plugin_dialog.direct_entry_edit.isVisible()
    assert plugin_dialog.direct_entry_btn.isVisible()


def test_constructor_visible_widgets(plugin_dialog_constructor):
    assert not plugin_dialog_constructor.direct_entry_edit.isVisible()
    assert not plugin_dialog_constructor.direct_entry_btn.isVisible()
