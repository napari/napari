from typing import TYPE_CHECKING

from ...utils.translations import trans
from ..dialogs.qt_about import QtAbout
from ._util import NapariMenu, populate_menu

if TYPE_CHECKING:
    from ..qt_main_window import Window


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
        try:
            from napari_error_monitor import _ask_opt_in, _settings_path

            def _show_dialog():
                _settings_path().unlink(missing_ok=True)
                _ask_opt_in()

            ACTIONS.append(
                {
                    'text': trans._('Bug reporting opt in/out...'),
                    'slot': _show_dialog,
                }
            )
        except ImportError:
            pass

        populate_menu(self, ACTIONS)
