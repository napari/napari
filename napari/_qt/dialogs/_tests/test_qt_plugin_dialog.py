from typing import Generator, Optional, Tuple

import pytest
from npe2 import PackageMetadata

from napari._qt.dialogs import qt_plugin_dialog


def _iter_napari_hub_or_pypi_plugin_info(
    conda_forge: bool = True,
) -> Generator[Tuple[Optional[PackageMetadata], bool], None, None]:
    """Mock the hub and pypi methods to collect available plugins.

    This will mock `napari.plugins.hub.iter_hub_plugin_info` for napari-hub,
    and `napari.plugins.pypi.iter_napari_plugin_info` for pypi.

    It will return two fake plugins that will populate the available plugins
    list (the bottom one). The first plugin will not be available on
    conda-forge so will be greyed out ("test-name-0"). The second plugin will
    be available on conda-forge so will be enabled ("test-name-1").
    """
    # This mock `base_data`` will be the same for both fake plugins.
    base_data = {
        "metadata_version": "1.0",
        "version": "0.1.0",
        "summary": "some test package",
        "home_page": "http://napari.org",
        "author": "test author",
        "license": "UNKNOWN",
    }
    for i in range(2):
        yield PackageMetadata(name=f"test-name-{i}", **base_data), bool(i), [
            '1',
            '2',
        ], ['3', '4.5']


@pytest.fixture
def plugin_dialog(qtbot, monkeypatch):
    """Fixture that provides a plugin dialog for a normal napari install."""
    for method_name in ["iter_hub_plugin_info", "iter_napari_plugin_info"]:
        monkeypatch.setattr(
            qt_plugin_dialog,
            method_name,
            _iter_napari_hub_or_pypi_plugin_info,
        )

    # This is patching `napari.utils.misc.running_as_constructor_app` function
    # to mock a normal napari install.
    monkeypatch.setattr(
        qt_plugin_dialog,
        "running_as_constructor_app",
        lambda: False,
    )

    widget = qt_plugin_dialog.QtPluginDialog()
    widget.show()
    qtbot.wait(300)
    qtbot.add_widget(widget)
    return widget


@pytest.fixture
def plugin_dialog_constructor(qtbot, monkeypatch):
    """
    Fixture that provides a plugin dialog for a constructor based install.
    """
    for method_name in ["iter_hub_plugin_info", "iter_napari_plugin_info"]:
        monkeypatch.setattr(
            qt_plugin_dialog,
            method_name,
            _iter_napari_hub_or_pypi_plugin_info,
        )

    # This is patching `napari.utils.misc.running_as_constructor_app` function
    # to mock a constructor based install.
    monkeypatch.setattr(
        qt_plugin_dialog,
        "running_as_constructor_app",
        lambda: True,
    )
    widget = qt_plugin_dialog.QtPluginDialog()
    widget.show()
    qtbot.wait(300)
    qtbot.add_widget(widget)
    return widget


def test_filter_not_available_plugins(plugin_dialog_constructor):
    """
    Check that the plugins listed under available plugins are
    enabled and disabled accordingly.

    The first plugin ("test-name-0") is not available on conda-forge and
    should be disabled, and show a tooltip warning.

    The second plugin ("test-name-1") is available on conda-forge and
    should be enabled without the tooltip warning.
    """
    item = plugin_dialog_constructor.available_list.item(0)
    widget = plugin_dialog_constructor.available_list.itemWidget(item)
    if widget:
        assert not widget.action_button.isEnabled()
        assert widget.warning_tooltip.isVisible()

    item = plugin_dialog_constructor.available_list.item(1)
    widget = plugin_dialog_constructor.available_list.itemWidget(item)
    assert widget.action_button.isEnabled()
    assert not widget.warning_tooltip.isVisible()


def test_filter_available_plugins(plugin_dialog):
    """
    Test the dialog is correctly filtering plugins in the available plugins
    list (the bottom one).
    """
    plugin_dialog.filter("")
    assert plugin_dialog.available_list.count() == 2
    assert plugin_dialog.available_list._count_visible() == 2

    plugin_dialog.filter("no-match@123")
    assert plugin_dialog.available_list._count_visible() == 0

    plugin_dialog.filter("")
    plugin_dialog.filter("test-name-0")
    assert plugin_dialog.available_list._count_visible() == 1


def test_filter_installed_plugins(plugin_dialog):
    """
    Test the dialog is correctly filtering plugins in the installed plugins
    list (the top one).
    """
    plugin_dialog.filter("")
    assert plugin_dialog.installed_list._count_visible() >= 0

    plugin_dialog.filter("no-match@123")
    assert plugin_dialog.installed_list._count_visible() == 0


def test_visible_widgets(plugin_dialog):
    """
    Test that the direct entry button and textbox are visible for
    normal napari installs.
    """

    assert plugin_dialog.direct_entry_edit.isVisible()
    assert plugin_dialog.direct_entry_btn.isVisible()


def test_constructor_visible_widgets(plugin_dialog_constructor):
    """
    Test that the direct entry button and textbox are hidden for
    constructor based napari installs.
    """
    assert not plugin_dialog_constructor.direct_entry_edit.isVisible()
    assert not plugin_dialog_constructor.direct_entry_btn.isVisible()


def test_version_dropdown(plugin_dialog):
    """
    Test that when the source drop down is changed, it displays the other versions properly.
    """
    plugin_dialog.available_list.item(
        1
    ).widget.source_choice_dropdown.setCurrentIndex(1)

    assert (
        plugin_dialog.available_list.item(
            1
        ).widget.version_choice_dropdown.currentText()
        == '4.5'
    )
