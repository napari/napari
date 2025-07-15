from app_model.types import (
    Action,
    SubmenuItem,
)

from napari._app_model.constants import MenuGroup, MenuId
from napari.components import ViewerModel
from napari.utils.translations import trans

FILE_SUBMENUS = [
    (
        MenuId.MENUBAR_FILE,
        SubmenuItem(
            submenu=MenuId.FILE_NEW_LAYER,
            title=trans._('New Layer'),
            group=MenuGroup.NAVIGATION,
            order=0,
        ),
    ),
    (
        MenuId.MENUBAR_FILE,
        SubmenuItem(
            submenu=MenuId.FILE_OPEN_WITH_PLUGIN,
            title=trans._('Open with Plugin'),
            group=MenuGroup.OPEN,
            order=99,
        ),
    ),
    (
        MenuId.MENUBAR_FILE,
        SubmenuItem(
            submenu=MenuId.FILE_SAMPLES,
            title=trans._('Open Sample'),
            group=MenuGroup.OPEN,
            order=100,
        ),
    ),
    (
        MenuId.MENUBAR_FILE,
        SubmenuItem(
            submenu=MenuId.FILE_IO_UTILITIES,
            title=trans._('IO Utilities'),
            group=MenuGroup.UTIL,
            order=101,
        ),
    ),
    (
        MenuId.MENUBAR_FILE,
        SubmenuItem(
            submenu=MenuId.FILE_ACQUIRE,
            title=trans._('Acquire'),
            group=MenuGroup.UTIL,
            order=101,
        ),
    ),
]


def add_new_points(viewer: 'ViewerModel') -> None:
    viewer.add_points(  # type: ignore[attr-defined]
        ndim=max(viewer.dims.ndim, 2),
        scale=viewer.layers.extent.step,
    )


def add_new_shapes(viewer: 'ViewerModel') -> None:
    viewer.add_shapes(  # type: ignore[attr-defined]
        ndim=max(viewer.dims.ndim, 2),
        scale=viewer.layers.extent.step,
    )


def new_labels(viewer: ViewerModel) -> None:
    viewer._new_labels()


def new_points(viewer: ViewerModel) -> None:
    add_new_points(viewer)


def new_shapes(viewer: ViewerModel) -> None:
    add_new_shapes(viewer)


FILE_ACTIONS: list[Action] = [
    Action(
        id='napari.window.file.new_layer.new_labels',
        title=trans._('Labels'),
        callback=new_labels,
        menus=[{'id': MenuId.FILE_NEW_LAYER, 'group': MenuGroup.NAVIGATION}],
    ),
    Action(
        id='napari.window.file.new_layer.new_points',
        title=trans._('Points'),
        callback=new_points,
        menus=[{'id': MenuId.FILE_NEW_LAYER, 'group': MenuGroup.NAVIGATION}],
    ),
    Action(
        id='napari.window.file.new_layer.new_shapes',
        title=trans._('Shapes'),
        callback=new_shapes,
        menus=[{'id': MenuId.FILE_NEW_LAYER, 'group': MenuGroup.NAVIGATION}],
    ),
]
