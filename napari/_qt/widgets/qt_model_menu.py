from __future__ import annotations

from typing import TYPE_CHECKING, List, Mapping, Optional

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QMenu

from ...utils.actions import commands_registry, menu_registry
from ...utils.actions._types import MenuItem, SubmenuItem

if TYPE_CHECKING:
    from qtpy.QtGui import QIcon

    from ...utils.actions import MenuId
    from ...utils.actions._types import Icon


def to_qicon(icon: Icon) -> QIcon:
    ...


class QtModelMenu(QMenu):
    def __init__(self, menu_id: MenuId, parent=None):
        super().__init__(parent)
        self._menu_id = menu_id
        self._submenu_item: Optional[SubmenuItem] = None
        self._submenus: List[QtModelMenu] = []
        self.rebuild()

    def rebuild(self):
        """Rebuild menu by looking up self._menu_id in menu_registry."""
        self.clear()

        groups = list(menu_registry.iter_menu_groups(self._menu_id))
        n_groups = len(groups)

        for n, group in enumerate(groups):
            for item in group:
                if isinstance(item, SubmenuItem):
                    sub = QtModelMenu(item.submenu, parent=self)
                    sub.setTitle(item.title)
                    if item.icon:
                        sub.setIcon(to_qicon(item.icon))
                    self.addMenu(sub)
                    self._submenus.append(sub)  # save pointer
                else:
                    action = self.addAction(item.command.title)
                    action.setData(item)
            if n < n_groups:
                self.addSeparator()

    def update_from_context(self, ctx: Mapping) -> None:
        """Update the enabled/visible state of each menu item with `ctx`.

        Parameters
        ----------
        ctx : Mapping
            A namepsace that will be used to `eval()` the `'enablement'` and
            `'when'` expressions provided for each action in the menu.
            *ALL variables used in these expressions must either be present in
            the `ctx` dict, or be builtins*.
        """
        for action in self.actions():
            item = action.data()
            if isinstance(item, MenuItem):
                action.setEnabled(
                    expr.eval(ctx)
                    if (expr := item.command.precondition)
                    else True
                )
                action.setVisible(
                    expr.eval(ctx) if (expr := item.when) else True
                )
            elif (menu := action.menu()) and isinstance(menu, QtModelMenu):
                menu.update_from_context(ctx)

    def exec(self, *args):
        if action := super().exec_(*args):
            if isinstance(item := action.data(), MenuItem):
                QTimer.singleShot(
                    0,
                    lambda: commands_registry.execute_command(item.command.id),
                )
