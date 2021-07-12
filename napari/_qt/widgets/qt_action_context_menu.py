from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Union

from qtpy.QtWidgets import QMenu

if TYPE_CHECKING:

    from ...layers._layer_actions import ActionDict


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

    def __init__(
        self, actions: Union[ActionDict, Sequence[ActionDict]], parent=None
    ):
        super().__init__(parent)
        if not isinstance(actions, (list, tuple)):
            actions = [actions]
        self._submenus = []
        self._build_menu(actions)

    # make menus behave like actions so we can add `enable_when` and stuff

    def setData(self, data):
        self._data = data

    def data(self):
        return self._data

    def update_from_context(self, ctx: dict) -> None:
        """Update the enabled/visible state of each menu item with `ctx`.

        `ctx` is a namepsace dict that will be used to `eval()` the
        `'enable_when'` and `'show_when'` expressions provided for each action
        in the menu. *ALL variables used in these expressions must either be
        present in the `ctx` dict, or be builtins*.
        """
        from itertools import chain

        for item in chain(self.actions(), self._submenus):
            d = item.data()
            if not d:
                continue
            enabled = eval(d['enable_when'], {}, ctx)
            item.setEnabled(enabled)
            visible = d.get("show_when")
            # let QMenus handle their own children
            if visible and not isinstance(item.parentWidget(), QMenu):
                item.setVisible(eval(visible, {}, ctx))

    def _build_menu(self, actions: Sequence[ActionDict]):
        # recursively build menu with submenus and sections
        for n, action in enumerate(actions):
            for val in action.values():
                if val.get('action_group'):
                    sub = QtActionContextMenu(
                        val.get('action_group'), parent=self
                    )
                    sub.setTitle(val['description'])
                    sub.setData(val)
                    self.addMenu(sub)
                    self._submenus.append(sub)  # save pointer
                else:
                    action = self.addAction(val['description'])
                    action.setData(val)
            if n < len(actions):
                self.addSeparator()
