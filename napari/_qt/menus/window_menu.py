from typing import TYPE_CHECKING

from napari._qt.menus._util import NapariMenu, populate_menu
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari._qt.qt_main_window import Window


class WindowMenu(NapariMenu):
    def __init__(self, window: 'Window'):
        super().__init__(trans._('&Window'), window._qt_window)
        ACTIONS = []
        populate_menu(self, ACTIONS)
