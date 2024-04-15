"""Actions related to the 'Help' menu that require Qt.

'Help' actions that do not require Qt should go in a new '_help_actions.py'
file within `napari/_app_model/actions/`.
"""

import sys

from app_model.types import Action, KeyBindingRule, KeyCode, KeyMod

from napari._app_model.constants import MenuGroup, MenuId
from napari._qt.dialogs.qt_about import QtAbout
from napari._qt.qt_main_window import Window
from napari.utils.translations import trans

try:
    from napari_error_reporter import ask_opt_in
except ModuleNotFoundError:
    ask_opt_in = None


def _show_about(window: Window):
    QtAbout.showAbout(window._qt_window)


Q_HELP_ACTIONS: list[Action] = [
    Action(
        id='napari.window.help.info',
        title=trans._('‎napari Info'),
        callback=_show_about,
        menus=[{'id': MenuId.MENUBAR_HELP, 'group': MenuGroup.RENDER}],
        status_tip=trans._('About napari'),
        keybindings=[KeyBindingRule(primary=KeyMod.CtrlCmd | KeyCode.Slash)],
    ),
    Action(
        id='napari.window.help.about_macos',
        title=trans._('About napari'),
        callback=_show_about,
        menus=[
            {
                'id': MenuId.MENUBAR_HELP,
                'group': MenuGroup.RENDER,
                'when': sys.platform == 'darwin',
            }
        ],
        status_tip=trans._('About napari'),
    ),
]

if ask_opt_in is not None:
    Q_HELP_ACTIONS.append(
        Action(
            id='napari.window.help.bug_report_opt_in',
            title=trans._('Bug Reporting Opt In/Out...'),
            callback=lambda: ask_opt_in(force=True),
            menus=[{'id': MenuId.MENUBAR_HELP}],
        )
    )
