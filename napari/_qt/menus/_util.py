from typing import TYPE_CHECKING, Callable, List, Union

from qtpy.QtWidgets import QAction, QMenu

from ...utils.menus import ActionMenuItem, CheckableMenuItem, Menu
from ..utils import convert_keybinding_to_shortcut

if TYPE_CHECKING:
    from typing_extensions import TypedDict

    from ...utils.events import EventEmitter

    try:
        from qtpy.QtCore import SignalInstance
    except ImportError:
        from qtpy.QtCore import pyqtBoundSignal as SignalInstance

    class ActionDict(TypedDict):
        text: str
        # these are optional
        slot: Callable
        shortcut: str
        statusTip: str
        menuRole: QAction.MenuRole
        checkable: bool
        checked: bool
        check_on: Union[EventEmitter, SignalInstance]

    class MenuDict(TypedDict):
        menu: str
        # these are optional
        items: List[ActionDict]

    # note: TypedDict still doesn't have the concept of "optional keys"
    # so we add in generic `dict` for type checking.
    # see PEP655: https://peps.python.org/pep-0655/
    MenuItem = Union[MenuDict, ActionDict, dict]


def populate_menu(menu: QMenu, actions: List['MenuItem']):
    """Populate a QMenu from a declarative list of QAction dicts.

    Parameters
    ----------
    menu : QMenu
        the menu to populate
    actions : list of dict
        A list of dicts with one or more of the following keys

        **Required: One of "text" or "menu" MUST be present in the dict**
        text: str
            the name of the QAction to add
        menu: str
            if present, creates a submenu instead.  "menu" keys may also
            provide an "items" key to populate the menu.

        **Optional:**
        slot: callable
            a callback to call when the action is triggered
        shortcut: str
            a keyboard shortcut to trigger the actoin
        statusTip: str
            used for setStatusTip
        menuRole: QAction.MenuRole
            used for setMenuRole
        checkable: bool
            used for setCheckable
        checked: bool
            used for setChecked (only if `checkable` is provided and True)
        check_on: EventEmitter
            If provided, and `checkable` is True, this EventEmitter will be
            connected to action.setChecked:

            `dct['check_on'].connect(lambda e: action.setChecked(e.value))`
    """
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
                sub.setParent(menu)
            else:
                sub = menu.addMenu(sub)
            populate_menu(sub, ax.get("items", []))
            continue
        action: QAction = menu.addAction(ax['text'])
        if 'slot' in ax:
            action.triggered.connect(ax['slot'])
        action.setShortcut(ax.get('shortcut', ''))
        action.setStatusTip(ax.get('statusTip', ''))
        if 'menuRole' in ax:
            action.setMenuRole(ax['menuRole'])
        if ax.get("checkable"):
            action.setCheckable(True)
            action.setChecked(ax.get("checked", False))
            if 'check_on' in ax:
                emitter = ax['check_on']

                @emitter.connect
                def _setchecked(e, action=action):
                    action.setChecked(e.value if hasattr(e, 'value') else e)

        action.setData(ax)


def populate_menu_view(view: QMenu, model: Menu):
    """Populate a menu view given its model.

    Parameters
    ----------
    view : qtpy.QtWidgets.QMenu
        Qt view to populate.
    menu : napari.components.menu.Menu
        Menu model to populate from.
    """
    for child in model.children:
        if isinstance(model, Menu):
            if child.enabled:
                # in case of a submenu, recurse
                submenu: QMenu = view.addMenu(child.label)
                populate_menu_view(submenu, child)
            else:
                submenu_placeholder = view.addAction(child.label)
                submenu_placeholder.setStatusTip(child.description)
                submenu_placeholder.setEnabled(False)
        else:
            # common attributes
            action: QAction = view.addAction(child.label)
            action.setShortcut(
                convert_keybinding_to_shortcut(child.keybinding)
            )
            action.setStatusTip(child.description)
            action.setEnabled(child.enabled)

            if isinstance(child, ActionMenuItem):
                # add callback to menu item
                action.triggered.connect(child.action)
            elif isinstance(child, CheckableMenuItem):
                # initialize check functionality
                action.setCheckable(True)
                action.setChecked(child.checked)

                # add callbacks
                @action.triggered.connect
                def onCheck(checked):
                    child.checked = checked

                @child.events.checked.connect
                def on_checked():
                    with child.events.checked.blocker():
                        action.setChecked(child.checked)


class NapariMenu(QMenu):
    """
    Base napari menu class that provides action handling and clean up on
    close.
    """

    _INSTANCES: List['NapariMenu'] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._INSTANCES.append(self)

    def _destroy(self):
        """Clean up action data to avoid widget leaks."""
        for ax in self.actions():
            ax.setData(None)

            try:
                ax._destroy()
            except AttributeError:
                pass

        if self in self._INSTANCES:
            self._INSTANCES.remove(self)

    def update(self, event=None):
        """Update action enabled/disabled state based on action data."""
        for ax in self.actions():
            data = ax.data()
            if data:
                enabled_func = data.get('enabled', lambda event: True)
                ax.setEnabled(bool(enabled_func(event)))
