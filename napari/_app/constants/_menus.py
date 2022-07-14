"""All Menus that are available in the napari GUI are defined here.

Internally, prefer using the `MenuId` enum instead of the string literal.

SOME of these (but definitely not all) will be exposed as "contributable"
menus for plugins to contribute commands and submenu items to.
"""

from enum import Enum


class MenuId(str, Enum):
    """Id representing a menu somewhere in napari."""

    VIEW = 'napari/view'
    LAYERLIST_CONTEXT = 'napari/layers/context'
    LAYERS_CONVERT_DTYPE = 'napari/layers/convert_dtype'
    LAYERS_PROJECT = 'napari/layers/project'

    def __str__(self) -> str:
        return self.value


# XXX: the structure/usage pattern of this class may change in the future
class MenuGroup:
    class LAYERLIST_CONTEXT:
        NAVIGATION = 'navigation'
        CONVERSION = '1_conversion'
        SPLIT_MERGE = '5_split_merge'
        LINK = '9_link'
