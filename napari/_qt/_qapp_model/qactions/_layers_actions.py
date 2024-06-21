from app_model.types import Action, SubmenuItem

from napari._app_model.constants import MenuGroup, MenuId
from napari._qt.qt_viewer import QtViewer
from napari.utils.translations import trans

LAYERS_SUBMENUS = [
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_VISUALIZE,
            title=trans._('Visualize'),
            group=MenuGroup.NAVIGATION,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_ANNOTATE,
            title=trans._('Annotate'),
            group=MenuGroup.NAVIGATION,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_TRANSFORM,
            title=trans._('Transform'),
            group=MenuGroup.LAYERS.GEOMETRY,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_FILTER,
            title=trans._('Filter'),
            group=MenuGroup.LAYERS.GEOMETRY,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_MEASURE,
            title=trans._('Measure'),
            group=MenuGroup.LAYERS.GEOMETRY,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_REGISTER,
            title=trans._('Register'),
            group=MenuGroup.LAYERS.GENERATE,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_PROJECT,
            title=trans._('Project'),
            group=MenuGroup.LAYERS.GENERATE,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_SEGMENT,
            title=trans._('Segment'),
            group=MenuGroup.LAYERS.GENERATE,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_TRACK,
            title=trans._('Track'),
            group=MenuGroup.LAYERS.GENERATE,
        ),
    ),
    (
        MenuId.MENUBAR_LAYERS,
        SubmenuItem(
            submenu=MenuId.LAYERS_CLASSIFY,
            title=trans._('Classify'),
            group=MenuGroup.LAYERS.GENERATE,
        ),
    ),
]


def new_labels(qt_viewer: QtViewer):
    viewer = qt_viewer.viewer
    viewer._new_labels()


LAYERS_ACTIONS: list[Action] = [
    Action(
        id='napari.layers.new_labels',
        title=trans._('Labels'),
        callback=new_labels,
        menus=[{'id': MenuId.FILE_NEW_LAYER, 'group': MenuGroup.NAVIGATION}],
    ),
]
