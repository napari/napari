import platform
from typing import TYPE_CHECKING

from ...utils.translations import trans
from ._util import NapariMenu, populate_menu

if TYPE_CHECKING:
    from ..qt_main_window import Window


class WindowMenu(NapariMenu):
    def __init__(self, window: 'Window'):
        super().__init__(trans._('&Window'), window._qt_window)
        ACTIONS = [
            {
                'when': platform.system() == "Darwin",
                'text': trans._('Minimize'),
                'slot': window._minimize,
                'shortcut': 'Ctrl+M',
                'statusTip': trans._('Minimize'),
            },
            {},
        ]
        populate_menu(self, ACTIONS)
