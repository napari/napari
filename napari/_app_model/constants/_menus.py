"""All Menus that are available anywhere in the napari GUI are defined here.

These might be menubar menus, context menus, or other menus.  They could
even be "toolbars", such as the set of mode buttons on the layer list.
A "menu" needn't just be a literal QMenu (though it usually is): it is better
thought of as a set of related commands.

Internally, prefer using the `MenuId` enum instead of the string literal.

SOME of these (but definitely not all) will be exposed as "contributable"
menus for plugins to contribute commands and submenu items to.
"""

from napari.utils.compat import StrEnum


class MenuId(StrEnum):
    """Id representing a menu somewhere in napari."""

    MENUBAR_FILE = 'napari/file'
    FILE_OPEN_WITH_PLUGIN = 'napari/file/open_with_plugin'
    FILE_SAMPLES = 'napari/file/samples'
    FILE_NEW_LAYER = 'napari/file/new_layer'
    FILE_IO_UTILITIES = 'napari/file/io_utilities'
    FILE_ACQUIRE = 'napari/file/acquire'

    MENUBAR_VIEW = 'napari/view'
    VIEW_AXES = 'napari/view/axes'
    VIEW_SCALEBAR = 'napari/view/scalebar'

    MENUBAR_LAYERS = 'napari/layers'
    LAYERS_VISUALIZE = 'napari/layers/visualize'
    LAYERS_ANNOTATE = 'napari/layers/annotate'

    LAYERS_DATA = 'napari/layers/data'
    LAYERS_LAYER_TYPE = 'napari/layers/layer_type'

    LAYERS_TRANSFORM = 'napari/layers/transform'
    LAYERS_MEASURE = 'napari/layers/measure'

    LAYERS_FILTER = 'napari/layers/filter'
    LAYERS_REGISTER = 'napari/layers/register'
    LAYERS_PROJECT = 'napari/layers/project'
    LAYERS_SEGMENT = 'napari/layers/segment'
    LAYERS_TRACK = 'napari/layers/track'
    LAYERS_CLASSIFY = 'napari/layers/classify'
    MENUBAR_WINDOW = 'napari/window'

    MENUBAR_PLUGINS = 'napari/plugins'

    MENUBAR_HELP = 'napari/help'

    MENUBAR_DEBUG = 'napari/debug'
    DEBUG_PERFORMANCE = 'napari/debug/performance_trace'

    LAYERLIST_CONTEXT = 'napari/layers/context'
    LAYERS_CONTEXT_CONVERT_DTYPE = 'napari/layers/context/convert_dtype'
    LAYERS_CONTEXT_PROJECT = 'napari/layers/contxt/project'
    LAYERS_CONTEXT_COPY_SPATIAL = 'napari/layers/context/copy_spatial'

    def __str__(self) -> str:
        return self.value

    @classmethod
    def contributables(cls) -> set['MenuId']:
        """Set of all menu ids that can be contributed to by plugins."""

        # TODO: add these to docs, with a lookup for what each menu is/does.
        _contributables = {
            cls.FILE_IO_UTILITIES,
            cls.FILE_ACQUIRE,
            cls.FILE_NEW_LAYER,
            cls.LAYERS_VISUALIZE,
            cls.LAYERS_ANNOTATE,
            cls.LAYERS_DATA,
            cls.LAYERS_LAYER_TYPE,
            cls.LAYERS_FILTER,
            cls.LAYERS_TRANSFORM,
            cls.LAYERS_MEASURE,
            cls.LAYERS_REGISTER,
            cls.LAYERS_PROJECT,
            cls.LAYERS_SEGMENT,
            cls.LAYERS_TRACK,
            cls.LAYERS_CLASSIFY,
        }
        return _contributables


# XXX: the structure/usage pattern of this class may change in the future
class MenuGroup:
    NAVIGATION = 'navigation'  # always the first group in any menu
    RENDER = '1_render'
    # View menu
    ZOOM = 'zoom'
    # Plugins menubar
    PLUGINS = '1_plugins'
    PLUGIN_MULTI_SUBMENU = '2_plugin_multi_submenu'
    PLUGIN_SINGLE_CONTRIBUTIONS = '3_plugin_contributions'
    # File menubar
    OPEN = '1_open'
    UTIL = '2_util'
    PREFERENCES = '3_preferences'
    SAVE = '4_save'
    CLOSE = '5_close'

    class LAYERLIST_CONTEXT:
        CONVERSION = '1_conversion'
        COPY_SPATIAL = '4_copy_spatial'
        SPLIT_MERGE = '5_split_merge'
        LINK = '9_link'

    class LAYERS:
        CONVERT = '1_convert'
        GEOMETRY = '2_geometry'
        GENERATE = '3_generate'


def is_menu_contributable(menu_id: str) -> bool:
    """Return True if the given menu_id is a menu that plugins can contribute to."""
    return (
        menu_id in MenuId.contributables()
        if menu_id.startswith('napari/')
        # TODO: this is intended to allow plugins to contribute to other plugins' menus but we
        # need to perform a more thorough check (probably not here though)
        else True
    )
