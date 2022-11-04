"""All Menus that are available anywhere in the napari GUI are defined here.

These might be menubar menus, context menus, or other menus.  They could
even be "toolbars", such as the set of mode buttons on the layer list.
A "menu" needn't just be a literal QMenu (though it usually is): it is better
thought of as a set of related commands.

Internally, prefer using the `MenuId` enum instead of the string literal.

SOME of these (but definitely not all) will be exposed as "contributable"
menus for plugins to contribute commands and submenu items to.
"""

from enum import Enum
from typing import Sequence, Set, Tuple

from app_model.types import SubmenuItem

from ...utils.translations import trans


class MenuId(str, Enum):
    """Id representing a menu somewhere in napari."""

    MENUBAR_VIEW = 'napari/view'
    VIEW_AXES = 'napari/view/axes'
    VIEW_SCALEBAR = 'napari/view/scalebar'

    MENUBAR_HELP = 'napari/help'

    LAYERLIST_CONTEXT = 'napari/layers/context'
    LAYERS_CONVERT_DTYPE = 'napari/layers/convert_dtype'
    LAYERS_PROJECT = 'napari/layers/project'

    def __str__(self) -> str:
        return self.value

    @classmethod
    def contributables(cls) -> Set['MenuId']:
        """Set of all menu ids that can be contributed to by plugins."""

        # TODO: add these to docs, with a lookup for what each menu is/does.
        _contributables = {
            cls.LAYERLIST_CONTEXT,
            cls.LAYERS_CONVERT_DTYPE,
            cls.LAYERS_PROJECT,
        }
        return _contributables

    @classmethod
    def sub_menus(cls) -> Sequence[Tuple['MenuId', SubmenuItem]]:
        """List of predefined submenu items to construct the default menu structure"""

        menu_id_to_sub_menus = {
            MenuId.LAYERLIST_CONTEXT: [
                {
                    'submenu': MenuId.LAYERS_CONVERT_DTYPE,
                    'title': trans._('Convert data type'),
                    'group': MenuGroup.LAYERLIST_CONTEXT.CONVERSION,
                    'order': None,
                },
                {
                    'submenu': MenuId.LAYERS_PROJECT,
                    'title': trans._('Projections'),
                    'group': MenuGroup.LAYERLIST_CONTEXT.SPLIT_MERGE,
                    'order': None,
                },
            ],
            MenuId.MENUBAR_VIEW: [
                {
                    'submenu': MenuId.VIEW_AXES,
                    'title': trans._('Axes'),
                },
                {
                    'submenu': MenuId.VIEW_SCALEBAR,
                    'title': trans._('Scale Bar'),
                },
            ],
        }

        return [
            (menu_id, SubmenuItem(**submenu))
            for menu_id, submenus in menu_id_to_sub_menus.items()
            for submenu in submenus
        ]


# XXX: the structure/usage pattern of this class may change in the future
class MenuGroup:
    NAVIGATION = 'navigation'  # always the first group in any menu
    RENDER = '1_render'

    class LAYERLIST_CONTEXT:
        CONVERSION = '1_conversion'
        SPLIT_MERGE = '5_split_merge'
        LINK = '9_link'


def is_menu_contributable(menu_id: str) -> bool:
    """Return True if the given menu_id is a menu that plugins can contribute to."""
    return (
        menu_id in MenuId.contributables()
        if menu_id.startswith("napari/")
        else True
    )
