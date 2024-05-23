from app_model.types import SubmenuItem

from napari._app_model.constants import MenuGroup, MenuId
from napari._app_model.context import LayerListSelectionContextKeys as LLSCK
from napari.utils.translations import trans

SUBMENUS = [
    (
        MenuId.LAYERLIST_CONTEXT,
        SubmenuItem(
            submenu=MenuId.LAYERS_CONVERT_DTYPE,
            title=trans._('Convert data type'),
            group=MenuGroup.LAYERLIST_CONTEXT.CONVERSION,
            order=None,
            enablement=LLSCK.all_selected_layers_labels,
        ),
    ),
    (
        MenuId.LAYERLIST_CONTEXT,
        SubmenuItem(
            submenu=MenuId.LAYERS_PROJECT,
            title=trans._('Projections'),
            group=MenuGroup.LAYERLIST_CONTEXT.SPLIT_MERGE,
            order=None,
            enablement=LLSCK.active_layer_is_image_3d,
        ),
    ),
    (
        MenuId.LAYERLIST_CONTEXT,
        SubmenuItem(
            submenu=MenuId.LAYERS_COPY_SPATIAL,
            title=trans._('Copy scale and transforms'),
            group=MenuGroup.LAYERLIST_CONTEXT.COPY_SPATIAL,
            order=None,
            enablement=(LLSCK.num_selected_layers == 1),
        ),
    ),
    (
        MenuId.MENUBAR_FILE,
        SubmenuItem(
            submenu=MenuId.FILE_OPEN_WITH_PLUGIN,
            title=trans._('Open with Plugin'),
            group=MenuGroup.NAVIGATION,
            order=99,
        ),
    ),
    (
        MenuId.MENUBAR_FILE,
        SubmenuItem(
            submenu=MenuId.FILE_SAMPLES,
            title=trans._('Open Sample'),
            group=MenuGroup.NAVIGATION,
            order=100,
        ),
    ),
    (
        MenuId.MENUBAR_VIEW,
        SubmenuItem(submenu=MenuId.VIEW_AXES, title=trans._('Axes')),
    ),
    (
        MenuId.MENUBAR_VIEW,
        SubmenuItem(submenu=MenuId.VIEW_SCALEBAR, title=trans._('Scale Bar')),
    ),
]
