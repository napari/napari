import importlib
from unittest import mock

import pytest
from app_model.types import MenuItem, SubmenuItem
from npe2 import DynamicPlugin
from qtpy.QtWidgets import QWidget

from napari._app_model import get_app_model
from napari._app_model.constants import MenuId
from napari._qt._qapp_model.qactions import _plugins, init_qactions
from napari._qt._qplugins._qnpe2 import _toggle_or_get_widget
from napari._tests.utils import skip_local_popups
from napari.plugins._tests.test_npe2 import mock_pm  # noqa: F401


class DummyWidget(QWidget):
    pass


@pytest.mark.skipif(
    not _plugins._plugin_manager_dialog_avail(),
    reason='`napari_plugin_manager` not available',
)
def test_plugin_manager_action(make_napari_viewer):
    """
    Test manage plugins installation action.

    The test is skipped in case `napari_plugin_manager` is not available
    """
    app = get_app_model()
    viewer = make_napari_viewer()

    with mock.patch(
        'napari_plugin_manager.qt_plugin_dialog.QtPluginDialog'
    ) as mock_plugin_dialog:
        app.commands.execute_command(
            'napari.window.plugins.plugin_install_dialog'
        )
    mock_plugin_dialog.assert_called_once_with(viewer.window._qt_window)


def test_plugin_errors_action(make_napari_viewer):
    """Test plugin errors action."""
    make_napari_viewer()
    app = get_app_model()

    with mock.patch(
        'napari._qt._qapp_model.qactions._plugins.QtPluginErrReporter.exec_'
    ) as mock_plugin_dialog:
        app.commands.execute_command(
            'napari.window.plugins.plugin_err_reporter'
        )
    mock_plugin_dialog.assert_called_once()


@skip_local_popups
def test_toggle_or_get_widget(
    make_napari_viewer,
    tmp_plugin: DynamicPlugin,
    qtbot,
):
    """Check `_toggle_or_get_widget` changes visibility correctly."""
    widget_name = 'Widget'
    plugin_name = 'tmp_plugin'
    full_name = 'Widget (Temp Plugin)'

    @tmp_plugin.contribute.widget(display_name=widget_name)
    def widget1():
        return DummyWidget()

    app = get_app_model()
    # Viewer needs to be visible
    viewer = make_napari_viewer(show=True)

    # Trigger the action, opening the widget: `Widget 1`
    app.commands.execute_command('tmp_plugin:Widget')
    widget = viewer.window._dock_widgets[full_name]
    # Widget takes some time to appear
    qtbot.waitUntil(widget.isVisible)
    assert widget.isVisible()

    # Define not visible callable
    def widget_not_visible():
        return not widget.isVisible()

    # Hide widget
    _toggle_or_get_widget(
        plugin=plugin_name,
        widget_name=widget_name,
        full_name=full_name,
    )
    qtbot.waitUntil(widget_not_visible)
    assert not widget.isVisible()

    # Make widget appear again
    _toggle_or_get_widget(
        plugin=plugin_name,
        widget_name=widget_name,
        full_name=full_name,
    )
    qtbot.waitUntil(widget.isVisible)
    assert widget.isVisible()


def test_plugin_single_widget_menu(
    make_napari_viewer, tmp_plugin: DynamicPlugin
):
    """Test single plugin widgets get added to the window menu correctly."""

    @tmp_plugin.contribute.widget(display_name='Widget 1')
    def widget1():
        return DummyWidget()

    app = get_app_model()
    viewer = make_napari_viewer()

    assert tmp_plugin.display_name == 'Temp Plugin'
    plugin_menu = app.menus.get_menu('napari/plugins')
    assert plugin_menu[0].command.title == 'Widget 1 (Temp Plugin)'
    assert len(viewer.window._dock_widgets) == 0
    assert 'tmp_plugin:Widget 1' in app.commands
    # trigger the action, opening the widget: `Widget 1`
    app.commands.execute_command('tmp_plugin:Widget 1')
    assert len(viewer.window._dock_widgets) == 1
    assert 'Widget 1 (Temp Plugin)' in viewer.window._dock_widgets


def test_plugin_multiple_widget_menu(
    make_napari_viewer,
    tmp_plugin: DynamicPlugin,
):
    """Check plugin with >1 widgets added with submenu and uses 'display_name'."""

    @tmp_plugin.contribute.widget(display_name='Widget 1')
    def widget1():
        return DummyWidget()

    @tmp_plugin.contribute.widget(display_name='Widget 2')
    def widget2():
        return DummyWidget()

    app = get_app_model()
    viewer = make_napari_viewer()

    assert tmp_plugin.display_name == 'Temp Plugin'
    plugin_menu = app.menus.get_menu('napari/plugins')
    assert plugin_menu[0].title == tmp_plugin.display_name
    plugin_submenu = app.menus.get_menu('napari/plugins/tmp_plugin')
    assert plugin_submenu[0].command.title == 'Widget 1'
    assert len(viewer.window._dock_widgets) == 0
    assert 'tmp_plugin:Widget 1' in app.commands
    # Trigger the action, opening the first widget: `Widget 1`
    app.commands.execute_command('tmp_plugin:Widget 1')
    assert len(viewer.window._dock_widgets) == 1
    assert 'Widget 1 (Temp Plugin)' in viewer.window._dock_widgets


def test_plugin_menu_plugin_state_change(
    make_napari_viewer,
    tmp_plugin: DynamicPlugin,
):
    """Check plugin menu items correct after a plugin changes state."""
    app = get_app_model()
    pm = tmp_plugin.plugin_manager

    # Register plugin q actions
    init_qactions()
    # Check only `Q_PLUGINS_ACTIONS` in plugin menu before any plugins registered
    plugins_menu = app.menus.get_menu(MenuId.MENUBAR_PLUGINS)
    assert len(plugins_menu) == len(_plugins.Q_PLUGINS_ACTIONS)

    @tmp_plugin.contribute.widget(display_name='Widget 1')
    def widget1():
        """Dummy widget."""

    @tmp_plugin.contribute.widget(display_name='Widget 2')
    def widget2():
        """Dummy widget."""

    # Configures `app`, registers actions and initializes plugins
    make_napari_viewer()

    plugins_menu = app.menus.get_menu(MenuId.MENUBAR_PLUGINS)
    assert len(plugins_menu) == len(_plugins.Q_PLUGINS_ACTIONS) + 1
    assert isinstance(plugins_menu[-1], SubmenuItem)
    assert plugins_menu[-1].title == tmp_plugin.display_name
    plugin_submenu = app.menus.get_menu(MenuId.MENUBAR_PLUGINS + '/tmp_plugin')
    assert len(plugin_submenu) == 2
    assert isinstance(plugin_submenu[0], MenuItem)
    assert plugin_submenu[0].command.title == 'Widget 1'
    assert 'tmp_plugin:Widget 1' in app.commands

    # Disable plugin
    pm.disable(tmp_plugin.name)
    with pytest.raises(KeyError):
        app.menus.get_menu(MenuId.MENUBAR_PLUGINS + '/tmp_plugin')
    assert 'tmp_plugin:Widget 1' not in app.commands

    # Enable plugin
    pm.enable(tmp_plugin.name)
    samples_sub_menu = app.menus.get_menu(
        MenuId.MENUBAR_PLUGINS + '/tmp_plugin'
    )
    assert len(samples_sub_menu) == 2
    assert 'tmp_plugin:Widget 1' in app.commands


def test_plugin_widget_checked(
    make_napari_viewer, qtbot, tmp_plugin: DynamicPlugin
):
    """Check widget toggling/hiding updates check mark correctly."""

    @tmp_plugin.contribute.widget(display_name='Widget')
    def widget_contrib():
        return DummyWidget()

    app = get_app_model()
    viewer = make_napari_viewer()

    assert 'tmp_plugin:Widget' in app.commands
    widget_action = viewer.window.plugins_menu.findAction('tmp_plugin:Widget')
    assert not widget_action.isChecked()
    # Trigger the action, opening the widget
    widget_action.trigger()
    assert widget_action.isChecked()
    assert 'Widget (Temp Plugin)' in viewer.window._dock_widgets
    widget = viewer.window._dock_widgets['Widget (Temp Plugin)']

    # Hide widget
    widget.title.hide_button.click()
    # Run `_on_about_to_show`, which is called on `aboutToShow`` event,
    # to update checked status
    viewer.window.plugins_menu._on_about_to_show()
    assert not widget_action.isChecked()

    # Trigger the action again to open widget and test item checked
    widget_action.trigger()
    assert widget_action.isChecked()


def test_import_plugin_manager():
    from napari_plugin_manager.qt_plugin_dialog import QtPluginDialog

    assert QtPluginDialog is not None


def test_plugin_manager(make_napari_viewer):
    """Test that the plugin manager is accessible from the viewer"""
    viewer = make_napari_viewer()
    assert _plugins._plugin_manager_dialog_avail()

    # Check plugin install action is visible
    plugin_install_action = viewer.window.plugins_menu.findAction(
        'napari.window.plugins.plugin_install_dialog',
    )
    assert plugin_install_action.isVisible()


def test_no_plugin_manager(monkeypatch, make_napari_viewer):
    """Test that the plugin manager menu item is hidden when not installed."""

    def mockreturn(*args):
        return None

    monkeypatch.setattr('importlib.util.find_spec', mockreturn)
    # We need to reload `_plugins` for the monkeypatching to work
    importlib.reload(_plugins)

    assert not _plugins._plugin_manager_dialog_avail()
    # Check plugin install action is not visible
    viewer = make_napari_viewer()
    plugin_install_action = viewer.window.plugins_menu.findAction(
        'napari.window.plugins.plugin_install_dialog',
    )
    assert not plugin_install_action.isVisible()


def test_plugins_menu_sorted(
    mock_pm,  # noqa: F811
    mock_app_model,
    tmp_plugin: DynamicPlugin,
):
    from napari._app_model import get_app_model
    from napari.plugins import _initialize_plugins

    # we make sure 'plugin-b' is registered first
    tmp_plugin2 = tmp_plugin.spawn(
        name='plugin-b', plugin_manager=mock_pm, register=True
    )
    tmp_plugin1 = tmp_plugin.spawn(
        name='plugin-a', plugin_manager=mock_pm, register=True
    )

    @tmp_plugin1.contribute.widget(display_name='Widget 1')
    def widget1(): ...

    @tmp_plugin1.contribute.widget(display_name='Widget 2')
    def widget2(): ...

    @tmp_plugin2.contribute.widget(display_name='Widget 1')
    def widget2_1(): ...

    @tmp_plugin2.contribute.widget(display_name='Widget 2')
    def widget2_2(): ...

    _initialize_plugins()
    plugins_menu = list(get_app_model().menus.get_menu('napari/plugins'))
    submenus = [item for item in plugins_menu if isinstance(item, SubmenuItem)]
    assert len(submenus) == 2
    assert submenus[0].title == 'plugin-a'
    assert submenus[1].title == 'plugin-b'
