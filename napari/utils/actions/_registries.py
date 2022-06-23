from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, overload

from psygnal import Signal

from ._types import Action, MenuItem

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

    from ._menus import MenuId
    from ._types import (
        CommandId,
        CommandRule,
        Icon,
        KeybindingRule,
        KeyCode,
        MenuRule,
        MenuRuleDict,
        TranslationOrStr,
    )

    DisposeCallable = Callable[[], None]


class CommandsRegistry:

    registered = Signal(str)
    _commands: Dict[CommandId, List[RegisteredCommand]] = {}
    __instance: Optional[CommandsRegistry] = None

    class RegisteredCommand(NamedTuple):
        id: str
        run: Callable

    @classmethod
    def instance(cls) -> CommandsRegistry:
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def register_command(
        self,
        id: CommandId,
        callback: Callable,
    ) -> DisposeCallable:
        commands = self._commands.setdefault(id, [])

        cmd = self.RegisteredCommand(id, run=callback)
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
        return f"<{self.__class__.__name__} ({len(self._commands)} commands)>"

    def __getitem__(self, id: CommandId) -> List[RegisteredCommand]:
        return self._commands[id]

    def execute(self, id: CommandId, *args, **kwargs):
        from .._injection import inject_napari_dependencies

        for cmd in self[id]:
            inject_napari_dependencies(cmd.run)(*args, **kwargs)


class KeybindingsRegistry:

    registered = Signal()
    _coreKeybindings: List[RegisteredKeyBinding] = []
    __instance: Optional[KeybindingsRegistry] = None

    class RegisteredKeyBinding(NamedTuple):
        keybinding: KeyCode
        command_id: CommandId
        weight: int
        when: Optional[context.Expr] = None

    @classmethod
    def instance(cls) -> KeybindingsRegistry:
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def register_keybinding_rule(
        self, id: CommandId, rule: KeybindingRule
    ) -> Optional[DisposeCallable]:
        if bound_keybinding := rule._bind_to_current_platform():
            entry = self.RegisteredKeyBinding(
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


class MenuRegistry:
    menus_changed = Signal(set)
    _menu_items: Dict[MenuId, List[MenuItem]] = {}
    _commands: Dict[CommandId, CommandRule] = {}
    __instance: Optional[MenuRegistry] = None

    @classmethod
    def instance(cls) -> MenuRegistry:
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def append_menu_items(
        self, items: Sequence[Tuple[MenuId, MenuItem]]
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

        if changed_ids:
            self.menus_changed.emit(changed_ids)

        return _dispose

    def add_commands(self, *commands: CommandRule):
        for command in commands:
            self._commands[command.id] = command
        # TODO: signal?

    def __iter__(self) -> Iterator[Tuple[MenuId, List[MenuItem]]]:
        yield from self._menu_items.items()

    def __contains__(self, id: object) -> bool:
        return id in self._menu_items


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
        CommandsRegistry.instance().register_command(action.id, action.run)
    ]

    # menu

    items = []
    for rule in action.menus or ():
        menu_item = MenuItem(
            command=action._reduce_to_rule(),
            when=rule.when,
            group=rule.group,
            order=rule.order,
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
