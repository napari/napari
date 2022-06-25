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
IconCode = NewType("IconCode", str)

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
    """Icons used to represent commands, or submenus.

    May provide both a light and dark variant.  If only one is provided, it is used
    in all theme types.
    """

    dark: Optional[IconCode] = Field(
        None,
        description="Icon path when a dark theme is used. These may be superqt "
        "fonticon keys, such as `fa5s.arrow_down`",
    )
    light: Optional[IconCode] = Field(
        None,
        description="Icon path when a light theme is used. These may be superqt "
        "fonticon keys, such as `fa5s.arrow_down`",
    )


class CommandRule(BaseModel):
    """Data representing a command and its presentation.

    Presentation of contributed commands depends on the containing menu. The Command
    Palette, for instance, prefixes commands with their category, allowing for easy
    grouping. However, the Command Palette doesn't show icons nor disabled commands.
    Menus, on the other hand, shows disabled items as grayed out, but don't show the
    category label.
    """

    id: CommandId = Field(
        ..., description="A globally unique identifier for the command."
    )
    title: TranslationOrStr = Field(
        ...,
        description="Title by which the command is represented in the UI.",
    )
    category: Optional[TranslationOrStr] = Field(
        None,
        description="(Optional) Category string by which the command may be grouped "
        "in the UI",
    )
    tooltip: Optional[TranslationOrStr] = None
    icon: Optional[Icon] = Field(
        None,
        description="(Optional) Icon used to represent this command, e.g. on buttons "
        "or in menus. These may be superqt fonticon keys, such as `fa5s.arrow_down`",
    )
    enablement: Optional[context.Expr] = Field(
        None,
        description="(Optional) Condition which must be true to enable the command in "
        "the UI (menu and keybindings). Does not prevent executing the command by "
        "other means, like the `execute_command` API.",
    )
    short_title: Optional[TranslationOrStr] = Field(
        None,
        description="(Optional) Short title by which the command is represented in "
        "the UI. Menus pick either `title` or `short_title` depending on the context "
        "in which they show commands.",
    )
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


class _RegisteredKeyBinding(NamedTuple):
    """Internal object representing a fully registered keybinding."""

    keybinding: KeyCode  # the keycode to bind to
    command_id: CommandId  # the command to run
    weight: int  # the weight of the binding, for prioritization
    when: Optional[
        context.Expr
    ] = None  # condition which must be true to enable


# menus


class _MenuItemBase(BaseModel):
    """Data representing where and when a menu item should be shown."""

    when: Optional[context.Expr] = Field(
        None,
        description="(Optional) Condition which must be true to show the item.",
    )
    group: Optional[str] = Field(
        None,
        description="(Optional) Menu group to which this item should be added. Menu "
        "groups are strings (like `'1_cutandpaste'`) that napari provides for specific "
        "menus. But plugins may also create groups. 'navigation' is a special group "
        "that always appears at the top of a menu.  If not provided, the item is added "
        "in the last group of the menu.",
    )
    order: Optional[float] = Field(
        None,
        description="(Optional) Order of the item *within* its group. Note, order is "
        "not part of the plugin schema, plugins may provide it using the group key "
        "and the syntax 'group@order'.  If not provided, items are sorted by title.",
    )


class MenuRule(_MenuItemBase):
    """A MenuRule defines a menu location and conditions for presentation.

    It does not define an actual command. That is done in either `MenuItem` or `Action`.
    """

    id: MenuId = Field(..., description="Menu in which to place this item.")


class MenuItem(_MenuItemBase):
    """Combination of a Command and conditions for menu presentation.

    This object is mostly constructed by `register_action` right before menu item
    registration.
    """

    command: CommandRule = Field(
        ...,
        description="CommandRule to execute when this menu item is selected.",
    )
    alt: Optional[CommandRule] = Field(
        None,
        description="(Optional) Alternate command to execute when this menu item is "
        "selected, (e.g. when the Alt-key is held when opening the menu)",
    )


class SubmenuItem(_MenuItemBase):
    """Point to another Menu that will be displayed as a submenu."""

    submenu: MenuId = Field(..., description="Menu to insert as a submenu.")
    title: TranslationOrStr = Field(
        ..., description="Title of this submenu, shown in the UI."
    )
    icon: Optional[Icon] = Field(
        None,
        description="(Optional) Icon used to represent this submenu. "
        "These may be superqt fonticon keys, such as `fa5s.arrow_down`",
    )


# Actions, potential combination of all the above
class Action(CommandRule):
    """Callable object along with specific context, menu, keybindings logic.

    This is the "complete" representation of a command.  Including a pointer to the
    actual callable object, as well as any additional menu and keybinding rules.
    Most internal napari commands and menu items will be represented by Actions,
    and registered using `register_action`.
    """

    run: Callable = Field(
        ...,
        description="A function to call when the associated CommandId is executed.",
    )
    menus: Optional[List[MenuRule]] = Field(
        None,
        description="(Optional) Menus to which this action should be added.",
    )
    keybindings: Optional[List[KeybindingRule]] = Field(
        None,
        description="(Optional) Default keybinding(s) that will trigger this command.",
    )
    add_to_command_palette: bool = Field(
        True,
        description="Whether to add this command to the global Command Palette "
        "during registration.",
    )


class _RegisteredCommand:
    """Small object to represent a command in the CommandsRegistry.

    Only used internally by the CommandsRegistry.
    This helper class allows us to cache the dependency-injected variant of the command.
    As usual with `cached_property`, the cache can be cleard by deleting the attribute:
    `del cmd.run_injected`
    """

    def __init__(
        self, id: CommandId, title: TranslationOrStr, run: Callable
    ) -> None:
        self.id = id
        self.title = title
        self.run = run

    @cached_property
    def run_injected(self):
        from .._injection import inject_napari_dependencies

        return inject_napari_dependencies(self.run)
