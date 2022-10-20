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
        MenuId.MENUBAR_VIEW,
        SubmenuItem(submenu=MenuId.VIEW_AXES, title=trans._('Axes')),
    ),
    (
        MenuId.MENUBAR_VIEW,
        SubmenuItem(submenu=MenuId.VIEW_SCALEBAR, title=trans._('Scale Bar')),
    ),
    (
        MenuId.MENUBAR_TOOLS,
        SubmenuItem(submenu=MenuId.TOOLS_ACQUISITION, title=trans._('Acquisition')),
    ),
    (
        MenuId.MENUBAR_TOOLS,
        SubmenuItem(submenu=MenuId.TOOLS_FILTERS, title=trans._('Filters')),
    ),
    (
        MenuId.MENUBAR_TOOLS,
        SubmenuItem(submenu=MenuId.TOOLS_TRANSFORM, title=trans._('Transform')),
    ),
    (
        MenuId.MENUBAR_TOOLS,
        SubmenuItem(submenu=MenuId.TOOLS_UTILITIES, title=trans._('Utilities')),
    ),
    (
        MenuId.MENUBAR_TOOLS,
        SubmenuItem(submenu=MenuId.TOOLS_MEASUREMENT, title=trans._('Measurement')),
    ),    (
        MenuId.MENUBAR_TOOLS,
        SubmenuItem(submenu=MenuId.TOOLS_CLASSIFICATION, title=trans._('Classification')),
    ),    (
        MenuId.MENUBAR_TOOLS,
        SubmenuItem(submenu=MenuId.TOOLS_PROJECTION, title=trans._('Projection')),
    ),
    (
        MenuId.MENUBAR_TOOLS,
        SubmenuItem(submenu=MenuId.TOOLS_SEGMENTATION, title=trans._('Segmentation')),
    ),
    (
        MenuId.MENUBAR_TOOLS,
        SubmenuItem(submenu=MenuId.TOOLS_VISUALIZATION, title=trans._('Visualization')),
    ),
]
