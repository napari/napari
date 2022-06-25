from __future__ import annotations

from typing import TYPE_CHECKING, overload

from ._registries import CommandsRegistry, KeybindingsRegistry, MenuRegistry
from ._types import Action, MenuItem

if TYPE_CHECKING:
    from typing import Any, Callable, List, Literal, Optional, Union

    from napari.utils import context

    from ._types import (
        CommandHandler,
        CommandId,
        Icon,
        KeybindingRule,
        KeybindingRuleDict,
        MenuRule,
        MenuRuleDict,
        TranslationOrStr,
    )

    KeybindingRuleOrDict = Union[KeybindingRule, KeybindingRuleDict]
    MenuRuleOrDict = Union[MenuRule, MenuRuleDict]
    DisposeCallable = Callable[[], None]
    CommandDecorator = Callable[[CommandHandler], CommandHandler]


@overload
def register_action(id_or_action: Action) -> DisposeCallable:
    ...


@overload
def register_action(
    id_or_action: CommandId,
    title: TranslationOrStr,
    *,
    run: Literal[None] = None,
    category: Optional[TranslationOrStr] = None,
    tooltip: Optional[TranslationOrStr] = None,
    icon: Optional[Icon] = None,
    enablement: Optional[context.Expr] = None,
    menus: Optional[List[MenuRuleOrDict]] = None,
    keybindings: Optional[List[KeybindingRuleOrDict]] = None,
    add_to_command_palette: bool = True,
) -> CommandDecorator:
    ...


@overload
def register_action(
    id_or_action: CommandId,
    title: TranslationOrStr,
    *,
    run: CommandHandler,
    category: Optional[TranslationOrStr] = None,
    tooltip: Optional[TranslationOrStr] = None,
    icon: Optional[Icon] = None,
    enablement: Optional[context.Expr] = None,
    menus: Optional[List[MenuRuleOrDict]] = None,
    keybindings: Optional[List[KeybindingRuleOrDict]] = None,
    add_to_command_palette: bool = True,
) -> DisposeCallable:
    ...


def register_action(
    id_or_action: Union[CommandId, Action],
    title: Optional[TranslationOrStr] = None,
    *,
    run: Optional[CommandHandler] = None,
    category: Optional[TranslationOrStr] = None,
    tooltip: Optional[TranslationOrStr] = None,
    icon: Optional[Icon] = None,
    enablement: Optional[context.Expr] = None,
    menus: Optional[List[MenuRuleOrDict]] = None,
    keybindings: Optional[List[KeybindingRuleOrDict]] = None,
    add_to_command_palette: bool = True,
) -> Union[CommandDecorator, DisposeCallable]:
    """Register an action.

    An Action is the "complete" representation of a command.  The command is the
    function itself, and an action also includes information about where and whether
    it appears in menus and optional keybinding rules.

    see also docstrings for:

    - :class:`~napari.utils.actions._types.Action`
    - :class:`~napari.utils.actions._types.CommandRule`
    - :class:`~napari.utils.actions._types.MenuRule`
    - :class:`~napari.utils.actions._types.KeybindingRule`

    This function can be used directly or as a decorator:

    - When the first `id_or_action` argument is an `Action`, then all other arguments
      are ignored, the action object is registered directly, and a function that may be
      used to unregister the action is returned.
    - When the first `id_or_action` argument is a string, it is interpreted as the `id`
      of the command being registered, and `title` must then also be provided. If `run`
      is not provided, then a decorator is returned that can be used to decorate the
      callable that executes the command; otherwise the command is registered directly
      and a function that may be used to unregister the action is returned.

    Parameters
    ----------
    id_or_action : Union[CommandId, Action]
        Either a complete Action object or a string id of the command being registered.
        If an `Action` object is provided, then all other arguments are ignored.
    title : Optional[TranslationOrStr]
        Title by which the command is represented in the UI. Required when
        `id_or_action` is a string.
    run : Optional[CommandHandler]
        Callable object that executes this command, by default None. If not provided,
        a decorator is returned that can be used to decorate a function that executes
        this action.
    category : Optional[TranslationOrStr]
        Category string by which the command may be grouped in the UI, by default None
    tooltip : Optional[TranslationOrStr]
        Tooltip to show when hovered., by default None
    icon : Optional[Icon]
        :class:`~napari.urils.actions._types.Icon` used to represent this command,
        e.g. on buttons or in menus. by default None
    enablement : Optional[context.Expr]
        Condition which must be true to enable the command in in the UI,
        by default None
    menus : Optional[List[MenuRuleOrDict]]
        :class:`~napari.utils.actions._types.MenuRule` or `dicts` containing menu
        placements for this action, by default None
    keybindings : Optional[List[KeybindingRuleOrDict]]
        :class:`~napari.utils.actions._types.KeybindingRule` or `dicts` containing
        default keybindings for this action, by default None
    add_to_command_palette : bool
        Whether to adds this command to the Command Palette, by default True

    Returns
    -------
    Union[CommandDecorator, DisposeCallable]
        If `run` is not provided, then a decorator is returned.
        If `run` is provided, or `id_or_action` is an `Action` object, then a function
        that may be used to unregister the action is returned.

    Raises
    ------
    ValueError
        If `id_or_action` is a string and `title` is not provided.
    TypeError
        If `id_or_action` is not a string or an `Action` object.
    """
    if isinstance(id_or_action, Action):
        return _register_action_obj(id_or_action)
    if isinstance(id_or_action, str):
        if not title:
            raise ValueError("'title' is required when 'id' is a string")
        return _register_action_str(
            id=id_or_action,
            title=title,
            category=category,
            tooltip=tooltip,
            icon=icon,
            enablement=enablement,
            run=run,
            add_to_command_palette=add_to_command_palette,
            menus=menus,
            keybindings=keybindings,
        )
    raise TypeError("'id_or_action' must be a string or an Action")


def _register_action_str(
    **kwargs: Any,
) -> Union[CommandDecorator, DisposeCallable]:
    """Create and register an Action with a string id and title.

    Helper for `register_action()`.

    If `kwargs['run']` is a callable, a complete `Action` is created
    (thereby performing type validation and casting) and registered with the
    corresponding registries. Otherwise a decorator returned that can be used
    to decorate the callable that executes the action.
    """
    if callable(kwargs.get('run')):
        return _register_action_obj(Action(**kwargs))

    def decorator(command: CommandHandler, **k) -> CommandHandler:
        _register_action_obj(Action(**{**kwargs, **k, 'run': command}))
        return command

    decorator.__doc__ = (
        f"Decorate function as callback for command {kwargs['id']!r}"
    )
    return decorator


def _register_action_obj(action: Action) -> DisposeCallable:
    """Register an Action object. Return a function that unregisters the action.

    Helper for `register_action()`.
    """
    # command
    disposers = [
        CommandsRegistry.instance().register_command(
            action.id, action.run, action.title
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
        # TODO: dispose
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
