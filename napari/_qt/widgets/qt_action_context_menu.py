from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from qtpy.QtWidgets import QMenu

if TYPE_CHECKING:
    from qtpy.QtWidgets import QAction

    from ...layers._layer_actions import ContextAction


class QtActionContextMenu(QMenu):
    """Makes a QMenu for a dict of `ContextActions`.

    `ContextActions` are just dicts with the following keys:
        description: str - the text for the menu item
        action: Callable - a callback when the item is selected
        enabled_when: str - an expression that will be evaluated with the namespace
            of some context.  If True, the menu item is enabled.
        hide_when: str|None - an expression

    Parameters
    ----------
    actions : Dict[str, ContextAction]
        [description]
    parent : [type], optional
        [description], by default None
    """

    def __init__(self, actions: Dict[str, ContextAction], parent=None):
        super().__init__(parent)
        self._actions = actions
        self._menu_actions: Dict[str, QAction] = {}

        for name, d in actions.items():
            if not d:
                self.addSeparator()
            else:
                self._menu_actions[name] = self.addAction(d['description'])
                self._menu_actions[name].setData(d['action'])

    def _update(self, ctx: dict) -> None:
        for name, menu_item in self._menu_actions.items():
            d = self._actions[name]
            enabled = eval(d['when'], {}, ctx)
            menu_item.setEnabled(enabled)
            _hidewhen = d.get("hide_when")
            if _hidewhen:
                menu_item.setVisible(not eval(_hidewhen, {}, ctx))
