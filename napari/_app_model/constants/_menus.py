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


class MenuId(str, Enum):
    """Id representing a menu somewhere in napari."""

    MENUBAR_VIEW = 'napari/view'
    VIEW_AXES = 'napari/view/axes'
    VIEW_SCALEBAR = 'napari/view/scalebar'

    LAYERLIST_CONTEXT = 'napari/layers/context'
    LAYERS_CONVERT_DTYPE = 'napari/layers/convert_dtype'
    LAYERS_PROJECT = 'napari/layers/project'

    def __str__(self) -> str:
        return self.value


# XXX: the structure/usage pattern of this class may change in the future
class MenuGroup:
    NAVIGATION = 'navigation'  # always the first group in any menu

    class LAYERLIST_CONTEXT:
        CONVERSION = '1_conversion'
        SPLIT_MERGE = '5_split_merge'
        LINK = '9_link'
