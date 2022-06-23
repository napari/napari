from __future__ import annotations

import os
import sys
from typing import Callable, List, NewType, Optional, Union

from pydantic import BaseModel

from ...utils import context
from ...utils.translations import TranslationString
from ._menus import MenuId

WINDOWS = os.name == 'nt'
MACOS = sys.platform == 'darwin'
LINUX = sys.platform.startswith("linux")

TranslationOrStr = Union[TranslationString, str]
CommandId = NewType("CommandId", str)
KeyCode = NewType("KeyCode", str)


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


# menus


class _MenuItemBase(BaseModel):
    when: Optional[context.Expr] = None
    group: str = "navigation"
    order: Optional[int] = None


class MenuRule(_MenuItemBase):
    id: MenuId


class MenuItem(_MenuItemBase):
    command: CommandRule
    alt: Optional[CommandRule] = None

    class Config:
        extra = 'ignore'


# commands


class CommandRule(BaseModel):
    id: CommandId
    title: TranslationOrStr
    short_title: Optional[TranslationOrStr] = None
    category: Optional[TranslationOrStr] = None
    tooltip: Optional[TranslationOrStr] = None
    icon: Optional[Icon] = None
    source: Optional[str] = None
    precondition: Optional[context.Expr] = None
    toggled: Optional[context.Expr] = None


class Icon(BaseModel):
    dark: Optional[str] = None
    light: Optional[str] = None


# Actions, potential combination of all the above
class Action(CommandRule):
    run: Callable
    description: Optional[str] = None
    menus: Optional[List[MenuRule]] = None
    keybindings: Optional[List[KeybindingRule]] = None
    add_to_command_palette: bool = True


MenuItem.update_forward_refs()
