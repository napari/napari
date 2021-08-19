from typing import List

from qtpy.QtWidgets import QAction, QMenu


def populate_menu(menu: QMenu, actions: List[dict]):
    for ax in actions:
        if not ax:
            menu.addSeparator()
            continue
        if not ax.get("when", True):
            continue
        if 'menu' in ax:
            sub = ax['menu']
            if isinstance(sub, QMenu):
                menu.addMenu(sub)
            else:
                sub = menu.addMenu(sub)
            populate_menu(sub, ax.get("items", []))
            continue
        action: QAction = menu.addAction(ax['text'])
        if ax['slot']:
            action.triggered.connect(ax['slot'])
        action.setShortcut(ax.get('shortcut', ''))
        action.setStatusTip(ax.get('statusTip', ''))
        if 'menuRole' in ax:
            action.setMenuRole(ax['menuRole'])
        if ax.get("checkable"):
            action.setCheckable(True)
            action.setChecked(ax.get("checked", False))
        if 'check_on' in ax:
            ax['check_on'].connect(lambda e: action.setChecked(e.value))
