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
from typing import Set


class MenuId(str, Enum):
    """Id representing a menu somewhere in napari."""

    MENUBAR_VIEW = 'napari/view'
    VIEW_AXES = 'napari/view/axes'
    VIEW_SCALEBAR = 'napari/view/scalebar'

    LAYERLIST_CONTEXT = 'napari/layers/context'
    LAYERS_CONVERT_DTYPE = 'napari/layers/convert_dtype'
    LAYERS_PROJECT = 'napari/layers/project'

    MENUBAR_TOOLS = 'napari/tools'
    TOOLS_ACQUISITION = 'napari/tools/acquisition'
    TOOLS_CLASSIFICATION = 'napari/tools/classification'
    TOOLS_FILTERS = 'napari/tools/filters'
    TOOLS_MEASUREMENT = 'napari/tools/measurement'
    TOOLS_SEGMENTATION = 'napari/tools/segmentation'
    TOOLS_PROJECTION = 'napari/tools/projection'
    TOOLS_TRANSFORM = 'napari/tools/transform'
    TOOLS_UTILITIES = 'napari/tools/utilities'
    TOOLS_VISUALIZATION = 'napari/tools/visualization'

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
            cls.TOOLS_ACQUISITION,
            cls.TOOLS_CLASSIFICATION,
            cls.TOOLS_FILTERS,
            cls.TOOLS_MEASUREMENT,
            cls.TOOLS_SEGMENTATION,
            cls.TOOLS_PROJECTION,
            cls.TOOLS_TRANSFORM,
            cls.TOOLS_UTILITIES,
            cls.TOOLS_VISUALIZATION,
        }
        return _contributables


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
