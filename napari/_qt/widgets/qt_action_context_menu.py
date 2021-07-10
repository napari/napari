from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from qtpy.QtWidgets import QMenu

if TYPE_CHECKING:
    from qtpy.QtWidgets import QAction

    from ...layers._layer_actions import ActionOrSeparator


class QtActionContextMenu(QMenu):
    """Makes a QMenu for a dict of `ContextActions`.

    `ContextActions` are just dicts with the following keys:
        description: str - the text for the menu item
        action: Callable - a callback when the item is selected
        enable_when: str - an expression that will be evaluated with the
            namespace of some context.  If True, the menu item is enabled.
        show_when: str|None - an expression that will be evaluated with the
            namespace of some context.  If True, the menu item is visible.
            If no show_when key is provided, the menu item is visible.

    Parameters
    ----------
    actions : Dict[str, ContextAction]
        An (ordered) mapping of name -> `ContextActions`.  Menu items will be
        added in order of the keys in the mapping.  To add a separator to the
        menu, add any key with a empty dict (or other falsy value).  The key
        itself doesn't matter.
    parent : QWidget, optional
        Parent widget, by default None

    Examples
    --------

    Start with an actions dict to populate the menu:

    >>> ACTIONS = {
    ...     'add_one': {
    ...         'description': 'Add one',
    ...         'action': lambda x: x.append(1),
    ...         'enable_when': 'count == 0 and is_ready',
    ...     },
    ... }
    >>> menu = QtActionContextMenu(ACTIONS)

    call menu.update_from_context to update the menu state:

    >>> menu.update_from_context({'count': 0, 'is_ready': True})
    >>> menu._menu_actions['add_one'].isEnabled()
    True

    We directly created the dict above, but a mapping of
    {key -> callable(obj)} is a good way to (re)create context
    dicts for an object that changes over time, like `my_list`:

    >>> my_list = [42]
    >>> CONTEXT_KEYS = {
    ...     'count': lambda x: len(x),
    ...     'is_ready': lambda x: True,
    ... }
    >>> ctx = {k: v(my_list) for k, v in CONTEXT_KEYS.items()}
    >>> ctx
    {'count': 1, 'is_ready': True}

    Use the context dict to update the menu.  Here, because count != 0,
    `add_one` becomes disabled

    >>> menu.update_from_context(ctx)
    >>> menu._menu_actions['add_one'].isEnabled()
    False
    """

    def __init__(self, actions: Dict[str, ActionOrSeparator], parent=None):
        super().__init__(parent)
        self._actions = actions
        self._menu_actions: Dict[str, QAction] = {}

        for name, d in actions.items():
            if not d:
                self.addSeparator()
            else:
                self._menu_actions[name] = self.addAction(d['description'])
                self._menu_actions[name].setData(d['action'])

    def update_from_context(self, ctx: dict) -> None:
        """Update the enabled/visible state of each menu item with `ctx`.

        `ctx` is a namepsace dict that will be used to `eval()` the
        `'enable_when'` and `'show_when'` expressions provided for each action
        in the menu. *ALL variables used in these expressions must either be
        present in the `ctx` dict, or be builtins*.
        """
        for name, menu_item in self._menu_actions.items():
            d = self._actions[name]
            enabled = eval(d['enable_when'], {}, ctx)
            menu_item.setEnabled(enabled)
            visible = d.get("show_when")
            if visible:
                menu_item.setVisible(eval(visible, {}, ctx))
