from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, overload

from psygnal import Signal

from ...utils.translations import trans
from ._menus import MenuGroup, MenuId
from ._types import (
    Action,
    MenuItem,
    RegisteredCommand,
    RegisteredKeyBinding,
    SubmenuItem,
)
from ._util import MockFuture

if TYPE_CHECKING:
    from typing import (
        Any,
        Callable,
        Dict,
        Iterator,
        List,
        Literal,
        Optional,
        Sequence,
        Set,
        Tuple,
        Union,
    )

    from napari.utils import context

    from ._types import (
        CommandId,
        CommandRule,
        Icon,
        KeybindingRule,
        MenuRule,
        MenuRuleDict,
        TranslationOrStr,
    )

    DisposeCallable = Callable[[], None]


class CommandsRegistry:

    registered = Signal(str)
    _commands: Dict[CommandId, List[RegisteredCommand]] = {}
    __instance: Optional[CommandsRegistry] = None

    @classmethod
    def instance(cls) -> CommandsRegistry:
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def register_command(
        self,
        id: CommandId,
        title: TranslationOrStr,
        callback: Callable,
    ) -> DisposeCallable:
        commands = self._commands.setdefault(id, [])

        cmd = RegisteredCommand(id, title, callback)
        commands.insert(0, cmd)

        def _dispose():
            commands.remove(cmd)
            if not commands:
                del self._commands[id]

        self.registered.emit(id)
        return _dispose

    def __iter__(self) -> Iterator[Tuple[CommandId, List[RegisteredCommand]]]:
        yield from self._commands.items()

    def __contains__(self, id: str) -> bool:
        return id in self._commands

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"<{name} at {hex(id(self))} ({len(self._commands)} commands)>"

    def __getitem__(self, id: CommandId) -> List[RegisteredCommand]:
        return self._commands[id]

    def execute_command(
        self, id: CommandId, *args, execute_asychronously=False, **kwargs
    ) -> Future:
        """Execute a registered command

        Parameters
        ----------
        id : CommandId
            ID of the command to execute
        execute_asychronously : bool
            Whether to execute the command asynchronously in a thread,
            by default `False`.  Note that *regardless* of this setting,
            the return value will implement the `Future` API (so it's necessary)
            to call `result()` on the returned object.  Eventually, this will
            default to True, but we need to solve `ensure_main_thread` Qt threading
            issues first

        Returns
        -------
        Future: conconrent.futures.Future
            Future object containing the result of the command

        Raises
        ------
        KeyError
            If the command is not registered or has no callbacks.
        """
        if cmds := self[id]:
            # TODO: decide whether we'll ever have more than one command
            # and if so, how to handle it
            cmd = cmds[0].run_injected
        else:
            raise KeyError(f'Command "{id}" has no registered callbacks')

        if execute_asychronously:
            with ThreadPoolExecutor() as executor:
                return executor.submit(cmd, *args, **kwargs)
        else:
            try:
                return MockFuture(cmd(*args, **kwargs), None)
            except Exception as e:
                return MockFuture(None, e)

    def __str__(self) -> str:
        lines: list = []
        for id, cmds in self:
            lines.extend(f"{id!r:<32} -> {cmd.title!r}" for cmd in cmds)
        return "\n".join(lines)


class KeybindingsRegistry:

    registered = Signal()
    _coreKeybindings: List[RegisteredKeyBinding] = []
    __instance: Optional[KeybindingsRegistry] = None

    @classmethod
    def instance(cls) -> KeybindingsRegistry:
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def register_keybinding_rule(
        self, id: CommandId, rule: KeybindingRule
    ) -> Optional[DisposeCallable]:
        if bound_keybinding := rule._bind_to_current_platform():
            entry = RegisteredKeyBinding(
                keybinding=bound_keybinding,
                command_id=id,
                weight=rule.weight,
                when=rule.when,
            )
            self._coreKeybindings.append(entry)
            self.registered.emit()

            def _dispose():
                self._coreKeybindings.remove(entry)

            return _dispose
        return None  # pragma: no cover

    def __iter__(self) -> Iterator[RegisteredKeyBinding]:
        yield from self._coreKeybindings

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"<{name} at {hex(id(self))} ({len(self._coreKeybindings)} bindings)>"


class MenuRegistry:
    menus_changed = Signal(set)
    _menu_items: Dict[MenuId, List[MenuItem | SubmenuItem]] = {}
    _commands: Dict[CommandId, CommandRule] = {}
    __instance: Optional[MenuRegistry] = None

    @classmethod
    def instance(cls) -> MenuRegistry:
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def append_menu_items(
        self, items: Sequence[Tuple[MenuId, MenuItem | SubmenuItem]]
    ) -> DisposeCallable:
        changed_ids: Set[MenuId] = set()
        disposers = []

        for id, item in items:
            menu_list = self._menu_items.setdefault(id, [])
            menu_list.append(item)
            changed_ids.add(id)
            disposers.append(lambda: menu_list.remove(item))

        def _dispose():
            for disposer in disposers:
                disposer()
            for id in changed_ids:
                if not self._menu_items.get(id):
                    del self._menu_items[id]
            self.menus_changed.emit(changed_ids)

        if changed_ids:
            self.menus_changed.emit(changed_ids)

        return _dispose

    def add_commands(self, *commands: CommandRule):
        for command in commands:
            self._commands[command.id] = command
        # TODO: signal?

    def __iter__(
        self,
    ) -> Iterator[Tuple[MenuId, List[MenuItem | SubmenuItem]]]:
        yield from self._menu_items.items()

    def __contains__(self, id: object) -> bool:
        return id in self._menu_items

    def __getitem__(self, id: MenuId) -> List[MenuItem | SubmenuItem]:
        return self._menu_items[id]

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"<{name} at {hex(id(self))} ({len(self._menu_items)} menus)>"

    def __str__(self) -> str:
        return "\n".join(self._render())

    def _render(self) -> List[str]:
        """Return registered menu items as lines of strings."""
        lines = []

        for menu, children in self:
            lines.append(menu.value)

            branch = "  ├──"
            for group in _sorted_groups(children):
                first = next(iter(group))
                lines.append(f"  ├───────────{first.group}───────────────")
                for child in group:
                    if isinstance(child, MenuItem):
                        lines.append(
                            f"{branch} {child.command.title} ({child.command.id})"
                        )
                    else:
                        lines.extend(
                            [
                                f"{branch} {child.submenu.value}",
                                "  ├──  └── ...",
                            ]
                        )
            lines.append('')
        return lines

    def iter_menu_groups(
        self, menu_id: MenuId
    ) -> Iterator[List[MenuItem | SubmenuItem]]:
        yield from _sorted_groups(self[menu_id])


def _sorted_groups(
    items: List[MenuItem | SubmenuItem],
    group_sorter: Callable = lambda x: 0 if x == 'navigation' else 1,
    order_sorter: Callable = lambda x: getattr(x, 'order', '') or 0,
) -> Iterator[List[MenuItem | SubmenuItem]]:
    """Sort a list of menu items based on their .group and .order attributes."""
    groups: dict[Optional[str], List[MenuItem | SubmenuItem]] = {}
    for item in items:
        groups.setdefault(item.group, []).append(item)

    for group_id in sorted(groups, key=group_sorter):
        yield sorted(groups[group_id], key=order_sorter)


@overload
def register_action(
    id_or_action: str,
    title: TranslationOrStr,
    *,
    short_title: Optional[TranslationOrStr] = None,
    category: Optional[TranslationOrStr] = None,
    tooltip: Optional[TranslationOrStr] = None,
    icon: Optional[Icon] = None,
    precondition: Optional[context.Expr] = None,
    run: Literal[None] = None,
    add_to_command_palette: bool = True,
    menus: Optional[List[Union[MenuRule, MenuRuleDict]]] = None,
    keybindings: Optional[List[KeybindingRule]] = None,
) -> Callable:
    ...


@overload
def register_action(
    id_or_action: str,
    title: TranslationOrStr,
    *,
    short_title: Optional[TranslationOrStr] = None,
    category: Optional[TranslationOrStr] = None,
    tooltip: Optional[TranslationOrStr] = None,
    icon: Optional[Icon] = None,
    precondition: Optional[context.Expr] = None,
    run: Callable,
    add_to_command_palette: bool = True,
    menus: Optional[List[Union[MenuRule, MenuRuleDict]]] = None,
    keybindings: Optional[List[KeybindingRule]] = None,
) -> DisposeCallable:
    ...


@overload
def register_action(id_or_action: Action) -> DisposeCallable:
    ...


def register_action(
    id_or_action: Union[str, Action],
    title: Optional[TranslationOrStr] = None,
    *,
    short_title: Optional[TranslationOrStr] = None,
    category: Optional[TranslationOrStr] = None,
    tooltip: Optional[TranslationOrStr] = None,
    icon: Optional[Icon] = None,
    precondition: Optional[context.Expr] = None,
    run: Optional[Callable] = None,
    add_to_command_palette: bool = True,
    menus: Optional[List[Union[MenuRule, MenuRuleDict]]] = None,
    keybindings: Optional[List[KeybindingRule]] = None,
) -> Union[Callable, DisposeCallable, None]:
    """Register an action.

    Can be used as a function or as a decorator.

    When the first `id_or_action` argument is a string, it is the `id` of the command
    being registered, and `title` must also be provided.  If `run` is not provided,
    then a decorator is returned that can be used to decorate the callable that
    executes the command.

    When the first `id_or_action` argument is an `Action`, then all other arguments
    are ignored, and the action object is registered directly.

    Parameters
    ----------
    id_or_action : Union[str, Action]
        _description_
    title : Optional[TranslationOrStr], optional
        _description_, by default None
    short_title : Optional[TranslationOrStr], optional
        _description_, by default None
    category : Optional[TranslationOrStr], optional
        _description_, by default None
    tooltip : Optional[TranslationOrStr], optional
        _description_, by default None
    icon : Optional[Icon], optional
        _description_, by default None
    source : Optional[str], optional
        _description_, by default None
    precondition : Optional[context.Expr], optional
        _description_, by default None
    run : Optional[Callable], optional
        _description_, by default None
    add_to_command_palette : bool, optional
        _description_, by default True
    menus : Optional[List[Union[MenuRule, MenuRuleDict]]], optional
        _description_, by default None
    keybindings : Optional[List[KeybindingRule]], optional
        _description_, by default None
    description : Optional[str], optional
        _description_, by default None

    Returns
    -------
    Union[Callable, Action, None]
        _description_

    Raises
    ------
    ValueError
        _description_
    TypeError
        _description_
    """
    if isinstance(id_or_action, Action):
        return _register_action(id_or_action)
    if isinstance(id_or_action, str):
        if title is None:
            raise ValueError("'title' is required when 'id' is a string")
        _kwargs = locals().copy()
        _kwargs['id'] = _kwargs.pop("id_or_action")
        return _register_action_str(**_kwargs)
    raise TypeError("'id_or_action' must be a string or an Action")


def _register_action_str(
    **kwargs: Any,
) -> Union[Callable[[Callable], Callable], DisposeCallable]:
    """Create and register an Action with a string id and title.

    Helper for `register_action()`.

    If the `kwargs['run']` is not callable, a decorator is created and returned.
    Otherwise an action is created (thereby performing validation and casting)
    and registered.
    """
    if not callable(kwargs.get('run')):

        def decorator(callable: Callable, **k) -> Callable:
            _register_action(Action(**{**kwargs, **k, 'run': callable}))
            return callable

        decorator.__doc__ = (
            f"Decorate function as callback for command {kwargs['id']!r}"
        )
        return decorator
    return _register_action(Action(**kwargs))


def _register_action(action: Action) -> DisposeCallable:
    """Register an Action object.

    Helper for `register_action()`.
    """
    # command
    disposers = [
        CommandsRegistry.instance().register_command(
            action.id, action.title, action.run
        )
    ]

    # menu

    items = []
    for rule in action.menus or ():
        menu_item = MenuItem(
            command=action, when=rule.when, group=rule.group, order=rule.order
        )
        items.append((rule.id, menu_item))

    disposers.append(MenuRegistry.instance().append_menu_items(items))
    if action.add_to_command_palette:
        # TODO: dispose?
        MenuRegistry.instance().add_commands(action)

    # keybinding
    for keyb in action.keybindings or ():
        if _d := KeybindingsRegistry.instance().register_keybinding_rule(
            action.id, keyb
        ):
            disposers.append(_d)

    def _dispose():
        for d in disposers:
            d()

    return _dispose


def _register_submenus():
    MenuRegistry.instance().append_menu_items(
        [
            (
                MenuId.LAYERLIST_CONTEXT,
                SubmenuItem(
                    submenu=MenuId.LAYERS_CONVERT_DTYPE,
                    title=trans._('Convert data type'),
                    group=MenuGroup.LAYERLIST_CONTEXT.CONVERSION,
                    order=None,
                ),
            ),
            (
                MenuId.LAYERLIST_CONTEXT,
                SubmenuItem(
                    submenu=MenuId.LAYERS_PROJECT,
                    title=trans._('Projections'),
                    group=MenuGroup.LAYERLIST_CONTEXT.SPLIT_MERGE,
                    order=None,
                ),
            ),
        ]
    )


_register_submenus()

menu_registry = MenuRegistry.instance()
commands_registry = CommandsRegistry.instance()
keybindings_registry = KeybindingsRegistry.instance()
