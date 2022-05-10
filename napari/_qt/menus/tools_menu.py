from typing import TYPE_CHECKING

from ...plugins import _npe2
from ._util import NapariMenu, populate_menu_view

if TYPE_CHECKING:
    from ..qt_main_window import Window


class ToolsMenu(NapariMenu):
    def __init__(self, window: 'Window'):
        self._win = window
        super().__init__('&Tools', window._qt_window)
        self._build()

    def _build(self):
        self.clear()
        model = _npe2.build_tools_menu()
        populate_menu_view(self, model)
