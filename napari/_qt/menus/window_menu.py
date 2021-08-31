from typing import TYPE_CHECKING

from qtpy.QtWidgets import QMenu

from ...utils.translations import trans
from ._util import populate_menu

if TYPE_CHECKING:
    from ..qt_main_window import Window


class WindowMenu(QMenu):
    def __init__(self, window: 'Window'):
        super().__init__(trans._('&Window'), window._qt_window)
        ACTIONS = []
        populate_menu(self, ACTIONS)
