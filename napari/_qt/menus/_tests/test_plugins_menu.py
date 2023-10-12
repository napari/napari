import importlib

import pytest
from app_model.types import MenuItem, SubmenuItem
from npe2 import DynamicPlugin
from qtpy.QtWidgets import QWidget

from napari._app_model import get_app
from napari._app_model.constants import CommandId, MenuId
from napari._qt._qapp_model.qactions import _plugins, init_qactions


class DummyWidget(QWidget):
    pass


def test_plugin_single_widget_menu(
    make_napari_viewer, tmp_plugin: DynamicPlugin
):
    """Test single plugin widgets get added to the window menu correctly."""

    @tmp_plugin.contribute.widget(display_name='Widget 1')
    def widget1():
        return DummyWidget()

    app = get_app()
    viewer = make_napari_viewer()

    assert tmp_plugin.display_name == 'Temp Plugin'
    plugin_menu = app.menus.get_menu('napari/plugins')
    assert plugin_menu[0].command.title == 'Widget 1 (Temp Plugin)'
    # Now ensure that the actions are still correct
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

    app = get_app()
    viewer = make_napari_viewer()

    assert tmp_plugin.display_name == 'Temp Plugin'
    plugin_menu = app.menus.get_menu('napari/plugins')
    assert plugin_menu[0].title == tmp_plugin.display_name
    # Now ensure that the actions are still correct
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
    app = get_app()
    pm = tmp_plugin.plugin_manager

    # Register plugin q actions
    init_qactions()
    # Check only `Q_PLUGINS_ACTIONS` in plugin menu before any plugins registered
    plugins_menu = app.menus.get_menu(MenuId.MENUBAR_PLUGINS)
    assert len(plugins_menu) == len(_plugins.Q_PLUGINS_ACTIONS)

    @tmp_plugin.contribute.widget(display_name='Widget 1')
    def widget1():
        return DummyWidget()

    @tmp_plugin.contribute.widget(display_name='Widget 2')
    def widget2():
        return DummyWidget()

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


def test_plugin_widget_checked(make_napari_viewer, tmp_plugin: DynamicPlugin):
    """Check widget toggling updates check mark correctly."""

    @tmp_plugin.contribute.widget(display_name='Widget')
    def widget():
        return DummyWidget()

    app = get_app()
    viewer = make_napari_viewer()

    assert 'tmp_plugin:Widget' in app.commands
    widget_action = viewer.window.plugins_menu.findAction('tmp_plugin:Widget')
    # Trigger the action, opening the widget
    assert not widget_action.isChecked()
    widget_action.trigger()
    assert widget_action.isChecked()
    assert 'Widget (Temp Plugin)' in viewer.window._dock_widgets


def test_import_plugin_manager():
    from napari_plugin_manager.qt_plugin_dialog import QtPluginDialog

    assert QtPluginDialog is not None


def test_plugin_manager(make_napari_viewer):
    """Test that the plugin manager is accessible from the viewer"""
    viewer = make_napari_viewer()
    assert _plugins._plugin_manager_dialog_avail()

    # Check plugin install action is visible
    plugin_install_action = viewer.window.plugins_menu.findAction(
        CommandId.DLG_PLUGIN_INSTALL,
    )
    assert plugin_install_action.isVisible()


def test_no_plugin_manager(monkeypatch, make_napari_viewer):
    """Test that the plugin manager menu item is hidden when not installed."""

    def mockreturn(*args):
        return None

    monkeypatch.setattr("importlib.util.find_spec", mockreturn)
    # We need to reload `_plugins` for the monkeypatching to work
    importlib.reload(_plugins)

    assert not _plugins._plugin_manager_dialog_avail()
    # Check plugin install action is not visible
    viewer = make_napari_viewer()
    plugin_install_action = viewer.window.plugins_menu.findAction(
        CommandId.DLG_PLUGIN_INSTALL
    )
    assert not plugin_install_action.isVisible()
