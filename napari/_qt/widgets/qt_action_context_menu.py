from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
)

from qtpy.QtWidgets import QAction, QMenu

from ...utils.context._expressions import Expr

if TYPE_CHECKING:
    from ...layers._layer_actions import MenuItem, SubMenu


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
    >>> menu._get_action('add_one').isEnabled()
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
    >>> menu._get_action('add_one').isEnabled()
    False
    """

    def __init__(
        self, actions: Union[MenuItem, Sequence[MenuItem]], parent=None
    ):
        super().__init__(parent)
        if not isinstance(actions, Sequence):
            actions = [actions]
        self._submenus: List[QtActionContextMenu] = []
        self._build_menu(actions)

    # make menus behave like actions so we can add `enable_when` and stuff

    def setData(self, data):
        self._data = data

    def data(self):
        return self._data

    def update_from_context(self, ctx: Mapping) -> None:
        """Update the enabled/visible state of each menu item with `ctx`.

        `ctx` is a namepsace dict that will be used to `eval()` the
        `'enable_when'` and `'show_when'` expressions provided for each action
        in the menu. *ALL variables used in these expressions must either be
        present in the `ctx` dict, or be builtins*.
        """
        for item in self.actions():
            if item.menu() is not None:
                item = item.menu()
            d = item.data()
            if not d:
                continue
            enable = d['enable_when']
            if isinstance(enable, Expr):
                enable = enable.eval(ctx)
            item.setEnabled(bool(enable))
            # if it's a menu, iterate (but don't toggle visibility)
            if isinstance(item, QtActionContextMenu):
                if enable:
                    item.update_from_context(ctx)
            else:
                vis = d.get("show_when")
                if vis is not None:
                    item.setVisible(
                        bool(vis.eval(ctx) if isinstance(vis, Expr) else vis)
                    )

    def _build_menu(self, actions: Sequence[MenuItem]):
        """recursively build menu with submenus and sections.

        Parameters
        ----------
        actions : Sequence[MenuItem]
            A sequence of `MenuItem` dicts.
            see `layers._layer_actions.MenuItem` for details and keys
        """
        for n, action in enumerate(actions):
            for key, val in action.items():
                if val.get('action_group'):
                    val = cast('SubMenu', val)
                    sub = QtActionContextMenu(val['action_group'], parent=self)  # type: ignore
                    sub.setTitle(val['description'])
                    sub.setData({**val, 'key': key})
                    self.addMenu(sub)
                    self._submenus.append(sub)  # save pointer
                else:
                    axtn = self.addAction(val['description'])
                    axtn.setData({**val, 'key': key})
            if n < len(actions):
                self.addSeparator()

    def _get_action(self, key: str) -> Optional[QAction]:
        """Get a QAction by the key that provided the `MenuItem` in _build_menu."""
        for action in self.actions():
            data = action.data() or {}
            if data.get('key') == key:
                return action
