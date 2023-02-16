import importlib.metadata
from typing import Generator, Optional, Tuple
from unittest.mock import patch

import npe2
import pytest

import napari.plugins
from napari._qt.dialogs import qt_plugin_dialog
from napari._qt.dialogs.qt_package_installer import InstallerActions
from napari.plugins._tests.test_npe2 import mock_pm  # noqa


def _iter_napari_hub_or_pypi_plugin_info(
    conda_forge: bool = True,
) -> Generator[Tuple[Optional[npe2.PackageMetadata], bool], None, None]:
    """Mock the hub and pypi methods to collect available plugins.

    This will mock `napari.plugins.hub.iter_hub_plugin_info` for napari-hub,
    and `napari.plugins.npe2api.iter_napari_plugin_info` for pypi.

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
        yield npe2.PackageMetadata(name=f"test-name-{i}", **base_data), bool(
            i
        ), {
            "home_page": 'www.mywebsite.com',
            "pypi_versions": ['3'],
            "conda_versions": ['4.5'],
        }


@pytest.fixture
def plugin_dialog(qtbot, monkeypatch, mock_pm):  # noqa
    """Fixture that provides a plugin dialog for a normal napari install."""

    class PluginManagerMock:
        def instance(self):
            return PluginManagerInstanceMock()

    class PluginManagerInstanceMock:
        def __init__(self):
            self.plugins = ['test-name-0', 'test-name-1', 'my-plugin']

        def __iter__(self):

            yield from self.plugins

        def iter_manifests(self):
            yield from [mock_pm.get_manifest('my-plugin')]

        def is_disabled(self, name):
            return False

        def discover(self):
            return ['plugin']

        def enable(self, plugin):
            return True

        def disable(self, plugin):
            return True

    class WarnPopupMock:
        def __init__(self, text):
            return None

        def exec_(self):
            return None

        def move(self, pos):
            return False

    def mock_metadata(name):
        meta = {
            'version': '0.1.0',
            'summary': '',
            'Home-page': '',
            'author': '',
            'license': '',
        }
        return meta

    class OldPluginManagerMock:
        def iter_available(self):
            return [('test-1', False, 'test-1')]

        def discover(self):
            return None

        def is_blocked(self, plugin):
            return False

        def set_blocked(self, plugin, blocked):
            return False

    for method_name in ["iter_hub_plugin_info", "iter_napari_plugin_info"]:
        monkeypatch.setattr(
            qt_plugin_dialog,
            method_name,
            _iter_napari_hub_or_pypi_plugin_info,
        )

    monkeypatch.setattr(qt_plugin_dialog, 'WarnPopup', WarnPopupMock)

    # This is patching `napari.utils.misc.running_as_constructor_app` function
    # to mock a normal napari install.
    monkeypatch.setattr(
        qt_plugin_dialog,
        "running_as_constructor_app",
        lambda: False,
    )

    monkeypatch.setattr(
        napari.plugins, 'plugin_manager', OldPluginManagerMock()
    )

    monkeypatch.setattr(importlib.metadata, 'metadata', mock_metadata)

    monkeypatch.setattr(npe2, 'PluginManager', PluginManagerMock())

    widget = qt_plugin_dialog.QtPluginDialog()
    widget.show()
    qtbot.wait(300)
    qtbot.add_widget(widget)
    yield widget
    widget.hide()
    widget._add_items_timer.stop()
    assert not widget._add_items_timer.isActive()


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
    yield widget
    widget.hide()
    widget._add_items_timer.stop()


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


def test_plugin_list_item(plugin_dialog):

    assert plugin_dialog.installed_list._count_visible() == 2


def test_plugin_list_handle_action(plugin_dialog, qtbot):
    item = plugin_dialog.installed_list.item(0)

    plugin_dialog.installed_list.handle_action(
        item, 'test-name-1', InstallerActions.INSTALL, update=True
    )

    with patch.object(qt_plugin_dialog.WarnPopup, "exec_") as mock:
        plugin_dialog.installed_list.handle_action(
            item, 'test-name-1', InstallerActions.UNINSTALL, update=False
        )
        assert mock.called

    item = plugin_dialog.available_list.item(0)
    plugin_dialog.available_list.handle_action(
        item,
        'test-name-1',
        InstallerActions.INSTALL,
        update=False,
        version='3',
    )

    plugin_dialog.available_list.handle_action(
        item, 'test-name-1', InstallerActions.CANCEL, update=False, version='3'
    )

    # Wait for refresh timer, state and worker to be done
    qtbot.waitUntil(
        lambda: not plugin_dialog._add_items_timer.isActive()
        and plugin_dialog.refresh_state == qt_plugin_dialog.RefreshState.DONE
    )
    qtbot.waitUntil(lambda: not plugin_dialog.worker.is_running)


def test_on_enabled_checkbox(plugin_dialog, qtbot):
    # checks npe2 lines
    item = plugin_dialog.installed_list.item(0)
    widget = plugin_dialog.installed_list.itemWidget(item)

    with qtbot.waitSignal(widget.enabled_checkbox.stateChanged, timeout=500):
        widget.enabled_checkbox.setChecked(False)

    # checks npe1 lines
    item = plugin_dialog.installed_list.item(1)
    widget = plugin_dialog.installed_list.itemWidget(item)

    with qtbot.waitSignal(widget.enabled_checkbox.stateChanged, timeout=500):
        widget.enabled_checkbox.setChecked(False)


def test_add_items_outdate(plugin_dialog):
    new_plugin = (
        npe2.PackageMetadata(name="my-plugin", version="0.3.0"),
        True,
        {
            "home_page": 'www.mywebsite.com',
            "pypi_versions": ['0.3.0'],
            "conda_versions": ['0.3.0'],
        },
    )
    plugin_dialog._plugin_data = [new_plugin]
    plugin_dialog._add_items()
