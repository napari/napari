from typing import TYPE_CHECKING

from qtpy.QtWidgets import QMenu

if TYPE_CHECKING:
    from ..qt_main_window import Window


def populate_qmenu_from_manifest(menu: QMenu, menu_key):
    from npe2 import execute_command, plugin_manager

    for item in plugin_manager.iter_menu(menu_key):
        if hasattr(item, 'submenu'):
            subm_contrib = plugin_manager.get_submenu(item.submenu)
            subm = menu.addMenu(subm_contrib.label)
            populate_qmenu_from_manifest(subm, subm_contrib.id)
        else:
            cmd = plugin_manager.get_command(item.command)
            action = menu.addAction(cmd.title)
            action.triggered.connect(lambda *_: execute_command(cmd.command))


class TestMenu(QMenu):
    KEY = 'test_menu'

    def __init__(self, window: 'Window'):
        super().__init__('&Test', window._qt_window)
        populate_qmenu_from_manifest(self, self.KEY)
