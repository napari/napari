from app_model.types import SubmenuItem

from ..utils.translations import trans
from .constants import MenuGroup, MenuId

SUBMENUS = [
    (
        MenuId.LAYERLIST_CONTEXT,
        SubmenuItem(
            submenu=MenuId.LAYERS_CONVERT_DTYPE,
            title=trans._('Convert data type'),
            group=MenuGroup.LAYERLIST_CONTEXT.CONVERSION,
            order=None,
        ),
    ),
    (
        MenuId.LAYERLIST_CONTEXT,
        SubmenuItem(
            submenu=MenuId.LAYERS_PROJECT,
            title=trans._('Projections'),
            group=MenuGroup.LAYERLIST_CONTEXT.SPLIT_MERGE,
            order=None,
        ),
    ),
    (
        MenuId.MENUBAR_FILE,
        SubmenuItem(
            submenu=MenuId.SAMPLES,
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
