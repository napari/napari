from typing import TYPE_CHECKING

from ...utils.translations import trans
from ..dialogs.qt_about import QtAbout
from ._util import NapariMenu, populate_menu

if TYPE_CHECKING:
    from ..qt_main_window import Window

try:
    from napari_error_reporter import ask_opt_in
except ModuleNotFoundError:
    ask_opt_in = None


class HelpMenu(NapariMenu):
    def __init__(self, window: 'Window'):
        super().__init__(trans._('&Help'), window._qt_window)
        ACTIONS = [
            {
                'text': trans._('napari Info'),
                'slot': lambda e: QtAbout.showAbout(window._qt_window),
                'shortcut': 'Ctrl+/',
                'statusTip': trans._('About napari'),
            }
        ]
        if ask_opt_in is not None:
            ACTIONS.append(
                {
                    'text': trans._('Bug reporting opt in/out...'),
                    'slot': lambda: ask_opt_in(force=True),
                }
            )

        populate_menu(self, ACTIONS)
