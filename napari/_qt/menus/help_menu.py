from typing import TYPE_CHECKING

from qtpy.QtWidgets import QMenu

from ...utils.translations import trans
from ..dialogs.qt_about import QtAbout
from ._util import populate_menu

if TYPE_CHECKING:
    from ..qt_main_window import Window


class HelpMenu(QMenu):
    def __init__(self, window: 'Window'):
        super().__init__(trans._('&Help'), window._qt_window)
        ACTIONS = [
            {
                'text': trans._('napari Info'),
                'slot': lambda e: QtAbout.showAbout(
                    window.qt_viewer, window._qt_window
                ),
                'shortcut': 'Ctrl+/',
                'statusTip': 'About napari',
            }
        ]
        populate_menu(self, ACTIONS)
