from ...utils.events import EventedList, NestableEventedList


def create_view(obj, parent=None):
    from . import QtListView, QtNodeTreeView

    if isinstance(obj, NestableEventedList):
        return QtNodeTreeView(obj, parent=parent)
    elif isinstance(obj, EventedList):
        return QtListView(obj, parent=parent)


def create_model(obj, parent=None):
    from . import QtListModel, QtNodeTreeModel

    if isinstance(obj, NestableEventedList):
        return QtNodeTreeModel(obj, parent=parent)
    elif isinstance(obj, EventedList):
        return QtListModel(obj, parent=parent)
