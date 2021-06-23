from typing import Dict

from qtpy.QtWidgets import QMenu

from ...layers._layer_actions import ContextAction


class QtActionContextMenu(QMenu):
    def __init__(self, actions: Dict[str, ContextAction]):
        super().__init__()
        self._actions = actions
        self._menu_actions = {}

        for name, d in actions.items():
            self._menu_actions[name] = self.addAction(d['description'])
            self._menu_actions[name].setData(d['action'])

    def _update(self, ctx: dict) -> None:
        for name, d in self._actions.items():
            menu_action = self._menu_actions[name]
            enabled = eval(d['when'], {}, ctx)
            menu_action.setEnabled(enabled)
            _hidewhen = d.get("hide_when")
            if _hidewhen:
                menu_action.setVisible(not eval(_hidewhen, {}, ctx))
