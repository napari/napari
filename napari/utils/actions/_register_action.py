from __future__ import annotations

from typing import TYPE_CHECKING, overload

from ._registries import CommandsRegistry, KeybindingsRegistry, MenuRegistry
from ._types import Action, MenuItem

if TYPE_CHECKING:
    from typing import Any, Callable, List, Literal, Optional, Union

    from napari.utils import context

    from ._types import (
        CommandHandler,
        Icon,
        KeybindingRule,
        MenuRule,
        MenuRuleDict,
        TranslationOrStr,
    )

    DisposeCallable = Callable[[], None]
    CommandDecorator = Callable[[CommandHandler], CommandHandler]


@overload
def register_action(id_or_action: Action) -> DisposeCallable:
    ...


@overload
def register_action(
    id_or_action: str,
    title: TranslationOrStr,
    *,
    category: Optional[TranslationOrStr] = None,
    tooltip: Optional[TranslationOrStr] = None,
    icon: Optional[Icon] = None,
    enablement: Optional[context.Expr] = None,
    run: Literal[None] = None,
    add_to_command_palette: bool = True,
    menus: Optional[List[Union[MenuRule, MenuRuleDict]]] = None,
    keybindings: Optional[List[KeybindingRule]] = None,
) -> CommandDecorator:
    ...


@overload
def register_action(
    id_or_action: str,
    title: TranslationOrStr,
    *,
    category: Optional[TranslationOrStr] = None,
    tooltip: Optional[TranslationOrStr] = None,
    icon: Optional[Icon] = None,
    enablement: Optional[context.Expr] = None,
    run: CommandHandler,
    add_to_command_palette: bool = True,
    menus: Optional[List[Union[MenuRule, MenuRuleDict]]] = None,
    keybindings: Optional[List[KeybindingRule]] = None,
) -> DisposeCallable:
    ...


def register_action(
    id_or_action: Union[str, Action],
    title: Optional[TranslationOrStr] = None,
    *,
    category: Optional[TranslationOrStr] = None,
    tooltip: Optional[TranslationOrStr] = None,
    icon: Optional[Icon] = None,
    enablement: Optional[context.Expr] = None,
    run: Optional[CommandHandler] = None,
    add_to_command_palette: bool = True,
    menus: Optional[List[Union[MenuRule, MenuRuleDict]]] = None,
    keybindings: Optional[List[KeybindingRule]] = None,
) -> Union[CommandDecorator, DisposeCallable]:
    """Register an action.

    Can be used as a function or as a decorator.

    When the first `id_or_action` argument is a string, it is the `id` of the command
    being registered, and `title` must also be provided.  If `run` is not provided,
    then a decorator is returned that can be used to decorate the callable that
    executes the command.

    When the first `id_or_action` argument is an `Action`, then all other arguments
    are ignored, and the action object is registered directly.
    """
    if isinstance(id_or_action, Action):
        return _register_action(id_or_action)
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
        return _register_action(Action(**kwargs))

    def decorator(command: CommandHandler, **k) -> CommandHandler:
        _register_action(Action(**{**kwargs, **k, 'run': command}))
        return command

    decorator.__doc__ = (
        f"Decorate function as callback for command {kwargs['id']!r}"
    )
    return decorator


def _register_action(action: Action) -> DisposeCallable:
    """Register an Action object.

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
