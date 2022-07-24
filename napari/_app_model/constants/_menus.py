"""All Menus that are available in the napari GUI are defined here.

Internally, prefer using the `MenuId` enum instead of the string literal.

SOME of these (but definitely not all) will be exposed as "contributable"
menus for plugins to contribute commands and submenu items to.
"""

from enum import Enum
from typing import Set


class MenuId(str, Enum):
    """Id representing a menu somewhere in napari."""

    LAYERLIST_CONTEXT = 'napari/layers/context'
    LAYERS_CONVERT_DTYPE = 'napari/layers/convert_dtype'
    LAYERS_PROJECT = 'napari/layers/project'

    def __str__(self) -> str:
        return self.value

    @classmethod
    @property
    def _contributables(cls) -> Set[str]:
        """Set of all menu ids that can be contributed to by plugins."""
        # TODO: add these to docs, with a lookup for what each menu is/does.
        return {MenuId.LAYERLIST_CONTEXT.value}


# XXX: the structure/usage pattern of this class may change in the future
class MenuGroup:
    NAVIGATION = 'navigation'  # always the first group in any menu

    class LAYERLIST_CONTEXT:
        CONVERSION = '1_conversion'
        SPLIT_MERGE = '5_split_merge'
        LINK = '9_link'


def is_menu_contributable(menu_id: str) -> bool:
    """Return True if the given menu_id is a menu that plugins can contribute to."""
    if menu_id.startswith("napari/"):
        return menu_id in MenuId._contributables  # type: ignore # (class property)
    return True
