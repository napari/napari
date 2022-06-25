from __future__ import annotations

import os
import sys
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    NamedTuple,
    NewType,
    Optional,
    TypedDict,
    Union,
)

from pydantic import BaseModel, Field

from ...utils import context
from ...utils.translations import TranslationString
from ._menus import MenuId

WINDOWS = os.name == 'nt'
MACOS = sys.platform == 'darwin'
LINUX = sys.platform.startswith("linux")

TranslationOrStr = Union[TranslationString, str]
CommandId = NewType("CommandId", str)
KeyCode = NewType("KeyCode", str)

if TYPE_CHECKING:

    # Typed dicts mimic the API of their pydantic counterparts.
    # Since pydantic allows you to pass in either an object or a dict,
    # This lets us use either anywhere, without losing typing support.
    # e.g. Union[MenuRuleDict, MenuRule]

    class MenuRuleDict(TypedDict, total=False):
        when: Optional[context.Expr]
        group: str
        order: Optional[float]
        id: MenuId

    class KeybindingRuleDict(TypedDict, total=False):
        primary: Optional[KeyCode]
        win: Optional[KeyCode]
        linux: Optional[KeyCode]
        mac: Optional[KeyCode]
        weight: int
        when: Optional[context.Expr]


# commands


class Icon(BaseModel):
    dark: Optional[str] = None
    light: Optional[str] = None


class CommandRule(BaseModel):
    id: CommandId = Field(
        ..., description="A globally unique identifier for the command."
    )
    title: TranslationOrStr = Field(
        ...,
        description="The title of the command. This will be used wherever a command is "
        "shown in the UI, such as in a menu, or a command palette.",
    )
    short_title: Optional[TranslationOrStr] = None
    category: Optional[TranslationOrStr] = None
    tooltip: Optional[TranslationOrStr] = None
    icon: Optional[Icon] = None
    precondition: Optional[context.Expr] = None
    # source: Optional[str] = None
    # toggled: Optional[context.Expr] = None


# keys


class KeybindingRule(BaseModel):
    primary: Optional[KeyCode] = None
    win: Optional[KeyCode] = None
    linux: Optional[KeyCode] = None
    mac: Optional[KeyCode] = None
    weight: int = 0
    when: Optional[context.Expr] = None

    def _bind_to_current_platform(self) -> Optional[KeyCode]:
        if WINDOWS and self.win:
            return self.win
        if MACOS and self.mac:
            return self.mac
        if LINUX and self.linux:
            return self.linux
        return self.primary


class RegisteredKeyBinding(NamedTuple):
    keybinding: KeyCode
    command_id: CommandId
    weight: int
    when: Optional[context.Expr] = None


# menus


class _MenuItemBase(BaseModel):
    when: Optional[context.Expr] = None
    group: Optional[str] = None
    order: Optional[float] = None


class MenuRule(_MenuItemBase):
    id: MenuId


class MenuItem(_MenuItemBase):
    command: CommandRule
    alt: Optional[CommandRule] = None


class SubmenuItem(_MenuItemBase):
    submenu: MenuId
    title: TranslationOrStr
    icon: Optional[Icon] = None


# Actions, potential combination of all the above
class Action(CommandRule):
    run: Callable
    menus: Optional[List[MenuRule]] = None
    keybindings: Optional[List[KeybindingRule]] = None
    add_to_command_palette: bool = True


class RegisteredCommand:
    def __init__(
        self, id: str, title: TranslationOrStr, run: Callable
    ) -> None:
        self.id = id
        self.title = title
        self.run = run

    @cached_property
    def run_injected(self):
        from .._injection import inject_napari_dependencies

        return inject_napari_dependencies(self.run)
