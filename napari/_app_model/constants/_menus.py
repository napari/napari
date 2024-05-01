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
    FILE_IO_UTILITIES = 'napari/file/io_utilities'

    MENUBAR_VIEW = 'napari/view'
    VIEW_AXES = 'napari/view/axes'
    VIEW_SCALEBAR = 'napari/view/scalebar'

    MENUBAR_LAYERS = 'napari/layers'
    LAYERS_VISUALIZE = 'napari/layers/visualize'
    LAYERS_NEW = 'napari/layers/new'
    LAYERS_EDIT = 'napari/layers/edit'
    LAYERS_EDIT_ANNOTATE = 'napari/layers/edit/annotate'
    LAYERS_EDIT_FILTER = 'napari/layers/edit/filter'
    LAYERS_EDIT_TRANSFORM = 'napari/layers/edit/transform'

    LAYERS_MEASURE = 'napari/layers/measure'

    LAYERS_REGISTRATION = 'napari/layers/registration'
    LAYERS_PROJECTION = 'napari/layers/projection'
    LAYERS_SEGMENTATION = 'napari/layers/segmentation'
    LAYERS_TRACKING = 'napari/layers/tracking'
    LAYERS_CLASSIFICATION = 'napari/layers/classification'

    MENUBAR_ACQUISITION = 'napari/acquisition'

    MENUBAR_PLUGINS = 'napari/plugins'

    MENUBAR_HELP = 'napari/help'

    LAYERLIST_CONTEXT = 'napari/layers/context'
    LAYERS_CONVERT_DTYPE = 'napari/layers/convert_dtype'
    LAYERS_PROJECT = 'napari/layers/project'
    LAYERS_COPY_SPATIAL = 'napari/layers/copy_spatial'

    def __str__(self) -> str:
        return self.value

    @classmethod
    def contributables(cls) -> set['MenuId']:
        """Set of all menu ids that can be contributed to by plugins."""

        # TODO: add these to docs, with a lookup for what each menu is/does.
        _contributables = {
            cls.FILE_IO_UTILITIES,
            cls.LAYERLIST_CONTEXT,
            cls.LAYERS_CONVERT_DTYPE,
            cls.LAYERS_PROJECT,
            cls.MENUBAR_ACQUISITION,
            cls.MENUBAR_LAYERS,
            cls.LAYERS_VISUALIZE,
            cls.LAYERS_NEW,
            cls.LAYERS_EDIT,
            cls.LAYERS_EDIT_ANNOTATE,
            cls.LAYERS_EDIT_FILTER,
            cls.LAYERS_EDIT_TRANSFORM,
            cls.LAYERS_MEASURE,
            cls.LAYERS_REGISTRATION,
            cls.LAYERS_PROJECTION,
            cls.LAYERS_SEGMENTATION,
            cls.LAYERS_TRACKING,
            cls.LAYERS_CLASSIFICATION,
        }
        return _contributables


# XXX: the structure/usage pattern of this class may change in the future
class MenuGroup:
    NAVIGATION = 'navigation'  # always the first group in any menu
    RENDER = '1_render'
    # Plugins menubar
    PLUGINS = '1_plugins'
    PLUGIN_MULTI_SUBMENU = '2_plugin_multi_submenu'
    PLUGIN_SINGLE_CONTRIBUTIONS = '3_plugin_contributions'
    # File menubar
    PREFERENCES = '2_preferences'
    SAVE = '3_save'
    CLOSE = '4_close'

    class LAYERLIST_CONTEXT:
        CONVERSION = '1_conversion'
        COPY_SPATIAL = '4_copy_spatial'
        SPLIT_MERGE = '5_split_merge'
        LINK = '9_link'

    class LAYERS:
        NEW = '1_new'
        EXISTING = '2_existing'
        GENERATE = '3_generate'
        PLUGINS = '4_plugins'


def is_menu_contributable(menu_id: str) -> bool:
    """Return True if the given menu_id is a menu that plugins can contribute to."""
    return (
        menu_id in MenuId.contributables()
        if menu_id.startswith('napari/')
        # TODO: this is intended to allow plugins to contribute to other plugins' menus but we
        # need to perform a more thorough check (probably not here though)
        else True
    )
