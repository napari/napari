"""Defines plugins menu q actions."""

from importlib.util import find_spec
from logging import getLogger

from app_model.types import Action

from napari._app_model.constants import CommandId, MenuGroup, MenuId
from napari._qt.dialogs.qt_plugin_report import QtPluginErrReporter
from napari._qt.qt_main_window import Window
from napari.utils.translations import trans

logger = getLogger(__name__)


def _plugin_manager_dialog_avail() -> bool:
    """Returns whether the plugin manager class is available."""

    plugin_dlg = find_spec('napari_plugin_manager')
    if plugin_dlg:
        return True
    # not available
    logger.debug('QtPluginDialog not available')
    return False


def _show_plugin_install_dialog(window: Window) -> None:
    """Show dialog that allows users to install and enable/disable plugins."""

    # TODO: Once menu contributions supported, `napari_plugin_manager` should be
    # amended to be a napari plugin and simply add this menu item itself.
    # This callback is only used when this package is available, thus we do not check
    from napari_plugin_manager.qt_plugin_dialog import QtPluginDialog

    QtPluginDialog(window._qt_window).exec_()


def _show_plugin_err_reporter(window: Window) -> None:
    """Show dialog that allows users to review and report plugin errors."""
    QtPluginErrReporter(parent=window._qt_window).exec_()  # type: ignore [attr-defined]


Q_PLUGINS_ACTIONS: list[Action] = [
    Action(
        id=CommandId.DLG_PLUGIN_INSTALL,
        title=CommandId.DLG_PLUGIN_INSTALL.command_title,
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
        title=CommandId.DLG_PLUGIN_ERR.command_title,
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
