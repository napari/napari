from app_model.types import Action

from napari._app_model.constants import MenuGroup, MenuId
from napari._qt.qt_viewer import QtViewer
from napari.utils.translations import trans


def new_labels(qt_viewer: QtViewer):
    viewer = qt_viewer.viewer
    viewer._new_labels()


LAYERS_ACTIONS: list[Action] = [
    Action(
        id='napari.layers.new_labels',
        title=trans._('Labels'),
        callback=new_labels,
        menus=[{'id': MenuId.LAYERS_NEW, 'group': MenuGroup.LAYERS.NEW}],
    ),
]
