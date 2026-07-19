from importlib.metadata import PackageNotFoundError, distribution

import pytest
from napari_plugin_manager.base_qt_package_installer import InstallerActions

pytestmark = pytest.mark.workflow

PACKAGE = 'napari-animation'

is_package_installed = True
try:
    distribution(PACKAGE)
except PackageNotFoundError:
    is_package_installed = False


@pytest.mark.xfail(
    is_package_installed,
    reason=f'{PACKAGE} plugin is already installed.',
)
def test_plugin_installation_uninstallation(
    make_napari_viewer, qtbot, monkeypatch
):
    """Test interaction between napari and the plugin manager dialog.

    The scenario evaluated consist in installing the `napari-animation`
    plugin through the plugin manager dialog, and then uninstalling it.
    """
    viewer = make_napari_viewer(show=False)

    # Trigger the plugin install dialog from the menu
    viewer.window.plugins_menu.findAction(
        'napari.window.plugins.plugin_install_dialog'
    ).trigger()
    plugin_manager_dialog = viewer.window._qt_window._plugin_manager_dialog

    # Check that napari-animation plugin is not installed
    init_installed_plugins = plugin_manager_dialog.installed_list.packages()
    assert PACKAGE not in init_installed_plugins

    # Add napari-animation to the direct entry text box and click the install button
    plugin_manager_dialog.direct_entry_edit.setText(PACKAGE)
    with qtbot.waitSignal(
        plugin_manager_dialog.installer.processFinished, timeout=60_000
    ):
        plugin_manager_dialog.direct_entry_btn.click()

    # Check that napari-animation plugin is installed
    installed_plugins = plugin_manager_dialog.installed_list.packages()
    assert PACKAGE in installed_plugins

    # Filter to show only the installed plugin
    plugin_manager_dialog.installed_list.filter(PACKAGE)

    # Remove the plugin to clean up after the test
    index = installed_plugins.index(PACKAGE)
    item = plugin_manager_dialog.installed_list.item(index)

    with qtbot.waitSignal(
        plugin_manager_dialog.installer.allFinished, timeout=30_000
    ):
        plugin_manager_dialog.installed_list.handle_action(
            item,
            PACKAGE,
            InstallerActions.UNINSTALL,
        )

    assert (
        init_installed_plugins
        == plugin_manager_dialog.installed_list.packages()
    )
