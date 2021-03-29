from .list.qt_list_model import QtListModel
from .list.qt_list_view import QtListView
from .tree.qt_tree_model import QtNodeTreeModel
from .tree.qt_tree_view import QtNodeTreeView

__all__ = ['QtListModel', 'QtListView', 'QtNodeTreeModel', 'QtNodeTreeView']


def create_view(obj):
    from ...utils.events import EventedList, NestableEventedList

    if isinstance(obj, NestableEventedList):
        return QtNodeTreeView(obj)
    elif isinstance(obj, EventedList):
        return QtListView(obj)
