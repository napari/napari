"""Defines plugins menu actions."""

from importlib.util import find_spec
from logging import getLogger
from typing import List, Optional

from app_model.types import Action

from napari._app_model.constants import CommandId, MenuGroup, MenuId
from napari._qt.dialogs.qt_plugin_report import QtPluginErrReporter
from napari._qt.qt_main_window import Window
from napari.utils.translations import trans

logger = getLogger(__name__)


def _plugin_manager_dialog_avail() -> Optional[type]:
    """Returns whether the plugin manager class is available."""

    plugin_dlg = find_spec('napari_plugin_manager')
    if plugin_dlg:
        return True
    # not available
    logger.debug("QtPluginDialog not available")
    return False


def _show_plugin_install_dialog(window: Window):
    """Show dialog that allows users to sort the call order of plugins."""

    # TODO: Register via plugin, once plugin menu contributions supported
    # This callback is only used when this package is available, thus we do not check
    from napari_plugin_manager.qt_plugin_dialog import QtPluginDialog

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
                'group': MenuGroup.PLUGINS,
                'order': 1,
                'when': _plugin_manager_dialog_avail(),
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
                'group': MenuGroup.PLUGINS,
                'order': 2,
            }
        ],
        callback=_show_plugin_err_reporter,
        status_tip=trans._(
            'Review stack traces for plugin exceptions and notify developers'
        ),
    ),
]
