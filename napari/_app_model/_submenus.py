from app_model.types import SubmenuItem

from napari._app_model.constants import MenuGroup, MenuId
from napari._app_model.context import LayerListContextKeys as LLCK
from napari.utils.translations import trans

SUBMENUS = [
    (
        MenuId.LAYERLIST_CONTEXT,
        SubmenuItem(
            submenu=MenuId.LAYERS_CONVERT_DTYPE,
            title=trans._('Convert data type'),
            group=MenuGroup.LAYERLIST_CONTEXT.CONVERSION,
            order=None,
            enablement=LLCK.all_selected_layers_labels,
        ),
    ),
    (
        MenuId.LAYERLIST_CONTEXT,
        SubmenuItem(
            submenu=MenuId.LAYERS_PROJECT,
            title=trans._('Projections'),
            group=MenuGroup.LAYERLIST_CONTEXT.SPLIT_MERGE,
            order=None,
            enablement=LLCK.active_layer_is_image_3d,
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
