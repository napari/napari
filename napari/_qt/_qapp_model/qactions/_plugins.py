"""Defines plugins menu actions."""

from logging import getLogger
from typing import List, Optional

from app_model.types import Action

from napari._app_model.constants import CommandId, MenuGroup, MenuId
from napari._qt.dialogs.qt_plugin_report import QtPluginErrReporter
from napari._qt.qt_main_window import Window
from napari.utils.translations import trans

logger = getLogger(__name__)


def _get_plugin_manager_dialog() -> Optional[type]:
    """Return the plugin manager class, if available."""
    try:
        # TODO: Register via plugin, once plugin menu contributions supported
        from napari_plugin_manager.qt_plugin_dialog import QtPluginDialog
    except ImportError as exc:
        logger.debug("QtPluginDialog not available", exc_info=exc)
        return None
    else:
        return QtPluginDialog


def _show_plugin_install_dialog(window: Window):
    """Show dialog that allows users to sort the call order of plugins."""
    # We don't check whether the class is not None, because this
    # function should only be connected in that case.
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
