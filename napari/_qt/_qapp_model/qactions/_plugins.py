"""Defines plugins menu actions."""

from typing import List

from app_model.types import Action

from ...._app_model.constants import CommandId, MenuId
from ....utils.translations import trans
from ...dialogs.qt_plugin_dialog import QtPluginDialog
from ...dialogs.qt_plugin_report import QtPluginErrReporter
from ...qt_main_window import Window


def _show_plugin_install_dialog(window: Window):
    """Show dialog that allows users to sort the call order of plugins."""
    QtPluginDialog(window._qt_window).exec_()


def _show_plugin_err_reporter(window: Window):
    """Show dialog that allows users to review and report plugin errors."""
    QtPluginErrReporter(parent=window._qt_window).exec_()


Q_PLUGINS_ACTIONS: List[Action] = [
    Action(
        id=CommandId.DLG_PLUGIN_INSTALL,
        title=CommandId.DLG_PLUGIN_INSTALL.title,
        menus=[
            {
                'id': MenuId.MENUBAR_PLUGINS,
                'group': '1_plugins',
                'order': 1,
            }
        ],
        callback=_show_plugin_install_dialog,
    ),
    Action(
        id=CommandId.DLG_PLUGIN_ERR,
        title=CommandId.DLG_PLUGIN_ERR.title,
        menus=[
            {
                'id': MenuId.MENUBAR_PLUGINS,
                'group': '1_plugins',
                'order': 2,
            }
        ],
        callback=_show_plugin_err_reporter,
        status_tip=trans._(
            'Review stack traces for plugin exceptions and notify developers'
        ),
    ),
]
